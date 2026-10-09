"""Engine-side `step_logits` producer — a `forward_hook` on the LM head.

Why this exists
---------------
`src/validation/logits_gate.py` needs TWO dumps with `step_logits` `[S, V]`: the reference
(`refdump.py --replay`) and **the engine**. Nothing in the repo produced the engine side, which
is what blocked L1.

Why a forward_hook and not a monkeypatch
----------------------------------------
`_regional_compilation` (`hpu_model_runner.py:5251-5300`) makes each `nn.ModuleList` child and
each `regional_compilation_layers_list` member (`[RMSNorm, VocabParallelEmbedding]`) its own
`torch.compile` region. A `forward_hook` on a **region module** runs in `nn.Module.__call__`
*outside* the compiled `forward`, so it adds no graph break and no specialization. This is the
mechanism `region_capture.py` uses and it is proven. The alternative — monkeypatching the
model's *inner in-region callables* — was measured to break `load_model`
(`TypeError: _build_compress_rope_cache() takes 4 positional arguments but 5 were given`,
2026-10-08) and must not be reused.

The contract (both sides RAW logits, fp32, `[S, V]`, shape-equal)
-----------------------------------------------------------------
  * engine    — ONE ROW PER FORWARD, taking the **last position**. Prefill (T=P) contributes
                row 0; each of the S-1 decode steps (T=1) contributes one row. Total S rows.
  * reference — `refdump.py --replay` runs `model(prompt + generated)` in ONE teacher-forced
                forward -> `lg` is `[P+S, V]`; the predicting rows are `lg[P-1 : P-1+S]`.
Raw logits on both sides, so NO log-softmax and no `SamplingParams(logprobs=...)` (vLLM's
sampling logprobs are log-softmax — a per-row scalar offset that would make `max_abs_delta`
meaningless while leaving `top1_match` intact).

Install: env-gated from `hpu_model_runner.py` next to `region_capture`, unset by default:
    DSV4_LOGITS_CAPTURE=<dir>   DSV4_LOGITS_CAPTURE_LABEL="..."
Output: `<dir>/step_logits_rank<rank>.pt` with `{"step_logits": [S, V] fp32, "label", "ids"}`.
"""
from __future__ import annotations

import atexit
import os

import torch

_CAP: dict = {"rows": [], "out": None, "label": "", "ids": None, "hooked": None}


def _rank() -> str:
    r = os.environ.get("RANK", os.environ.get("LOCAL_RANK"))
    if r is not None:
        return str(r)
    try:
        import torch.distributed as dist

        return str(dist.get_rank()) if dist.is_initialized() else "0"
    except Exception:  # noqa: BLE001
        return "0"


def _last_row(out):
    """The last-position logits from one head forward, for both plausible layouts."""
    if not torch.is_tensor(out):
        if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
            out = out[0]
        else:
            return None
    if out.dim() == 3:          # [B, T, V]
        return out[0, -1, :]
    if out.dim() == 2:          # [T, V]
        return out[-1, :]
    return None


def install(model, out_dir: str | None = None, label: str = ""):
    """Register the LM-head forward_hook. Idempotent. Returns the capture dict."""
    if _CAP["hooked"] is not None:
        return _CAP
    out_dir = out_dir or os.environ.get("DSV4_LOGITS_CAPTURE")
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    _CAP["out"] = out_dir
    _CAP["label"] = label or os.environ.get("DSV4_LOGITS_CAPTURE_LABEL", "engine step_logits")

    # WHY NOT A forward_hook ON THE LM HEAD: vLLM never calls the head as a module.
    # `LogitsProcessor._get_logits` uses `lm_head.quant_method.apply(...)` or
    # `torch.mm(flat, lm_head.weight.t(), ...)` directly (vllm/model_executor/layers/
    # logits_processor.py:106,127), so `nn.Module.__call__` is bypassed and a forward_hook
    # on it can never fire. Measured 2026-10-08: the hook installed on all 4 workers and
    # recorded 0 rows after a full prefill.
    #
    # The boundary that DOES work: `compute_logits` is a compiled REGION
    # (`_compile_methods` -> `_compile_region(self.model, 'compute_logits', ...)`, run at
    # hpu_model_runner.py:5250, i.e. BEFORE this install site at :5264) and the runner calls
    # it from its OWN Python (`logits = self.model.compute_logits(hidden_states)`,
    # hpu_model_runner.py:4010). Wrapping that entry point therefore runs outside the compiled
    # program -- the same property as a forward_hook on a region module, and it is a REGION
    # ENTRY POINT, not an inner in-region callable (the distinction that broke gap B).
    orig = model.compute_logits

    def wrapper(*a, **kw):
        out = orig(*a, **kw)
        if torch.is_tensor(out) and out.numel():
            row = out[-1, :] if out.dim() == 2 else out.reshape(-1, out.shape[-1])[-1, :]
            _CAP["rows"].append(row.detach().float().cpu().clone())
        return out

    model.compute_logits = wrapper
    _CAP["hooked"] = "compute_logits"
    atexit.register(_write)
    print(f"[logits_capture] installed on {_CAP['hooked']} -> {out_dir} "
          f"(label={_CAP['label']!r})", flush=True)
    return _CAP


def set_ids(ids):
    _CAP["ids"] = list(ids) if ids is not None else None


def _write():
    if not _CAP["rows"] or not _CAP["out"]:
        return
    path = os.path.join(_CAP["out"], f"step_logits_rank{_rank()}.pt")
    try:
        x = torch.stack(_CAP["rows"], dim=0)          # [S, V]
        torch.save({"step_logits": x, "label": _CAP["label"], "ids": _CAP["ids"]}, path)
        print(f"[logits_capture] wrote {path}: S={x.shape[0]} V={x.shape[1]} "
              f"dtype={x.dtype}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[logits_capture] write failed: {type(e).__name__}: {e}", flush=True)
