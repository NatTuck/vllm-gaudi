"""Capture the per-region (per-decoder-layer) outputs of a running engine.

Mission step 3. `src/validation/refdump.py` produces the independent reference at layer
boundaries; this module captures the engine's tensors at the *same* boundaries so
`src/validation/chain_gate.py` can diff them per region.

Why a `forward_hook` on the region modules is the right mechanism: `_regional_compilation`
(`hpu_model_runner.py:5272-5300`) makes each `nn.ModuleList` child and each `RMSNorm` /
`VocabParallelEmbedding` its own `torch.compile` region. A `forward_hook` on such a module runs
in `nn.Module.__call__` **after** the compiled `forward` returns, i.e. it reads the region's
output *outside* the compiled program. So the capture does not enter the graph, add a graph
break, or change any specialization — which is exactly the constraint the mission states
("capture is only possible at region boundaries without breaking the graph").

Cost: the hook copies one `[T, 4096]` bf16 tensor per region per forward, so this is a
**validation** run, not a perf run. The largest-T forward (the prefill) is kept and the capture
is written at process exit; decode steps (T=1) never overwrite it.

Enable with `DSV4_REGION_CAPTURE=<dir>` (unset by default -> nothing is installed).
"""
from __future__ import annotations

import atexit
import os

import torch

_CAP: dict = {"regions_in": {}, "regions_out": {}, "streams_in": {}, "layer_state": {},
              "best_T": -1, "ids": None, "label": "", "out": None}
_layers_ref: list = [None]   # the decoder-layer list, for the write trigger


def _pick_layer_list(model):
    """The `nn.ModuleList` holding the decoder layers (the most numerous one)."""
    best, best_n = None, 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > best_n:
            best, best_n = mod, len(mod)
    return best


def _flatten(out):
    """A decoder layer may return `hidden` or `(hidden, residual)` (DSv4 mHC)."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def install(model, out_dir: str | None = None, label: str = ""):
    """Register hooks on every decoder-layer region. Idempotent."""
    if _CAP["out"] is not None:
        return
    out_dir = out_dir or os.environ.get("DSV4_REGION_CAPTURE")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    _CAP["out"] = out_dir
    _CAP["label"] = label or os.environ.get("DSV4_REGION_CAPTURE_LABEL", "engine capture")
    want_len = int(os.environ.get("DSV4_REGION_CAPTURE_LEN", "0"))

    layers = _pick_layer_list(model)
    if layers is None or len(layers) == 0:
        print("[region_capture] no decoder-layer ModuleList found", flush=True)
        return

    def make_hook(idx):
        def hook(_module, args, output):
            # Finding 10/11 (2026-10-06): the decoder layer is called positionally as
            # ``(x, positions, input_ids, post_mix, res_mix, residual)`` (deepseek_v4.py:2501),
            # so ``args[5]`` is the *mHC residual state* -- the ``[T, hc_mult, H]`` tensor the
            # reference's ``hidden_streams`` corresponds to. ``args[0]`` (what ``regions_in``
            # holds) is the post-collapse ``[T, H]`` sublayer input, a different object; without
            # this the chain gate can only compare a collapse against a collapse, which cannot
            # distinguish "wrong boundary pairing" from "diverged streams".
            # The DSv4 *layer boundary* tensor is the mHC residual state, `[T, HC, H]` with
            # HC = 4 -- that is what the reference's `hidden_states[i]` exposes. The layer's
            # *output* is the post-collapse `[T, H]`, so both are captured: `regions_in` is the
            # boundary the reference dump can be diffed against, `regions_out` is a fallback.
            ti = _flatten(args[0]) if args else None
            res = args[5] if len(args) > 5 else None
            t = _flatten(output)
            if torch.is_tensor(ti) and int(ti.shape[0]) >= 1:
                T = int(ti.shape[0])
                if (not want_len or T >= want_len) and T >= _CAP["best_T"]:
                    _CAP["best_T"] = T
                    _CAP["regions_in"][idx] = ti.detach().to(torch.bfloat16).cpu().clone()
            if torch.is_tensor(res) and res.dim() == 3 and res.shape[0] >= 1:
                if (not want_len or int(res.shape[0]) >= want_len) \
                        and int(res.shape[0]) >= _CAP["best_T"]:
                    _CAP["streams_in"][idx] = res.detach().cpu()
            # The layer returns ``(x, residual, post_mix, res_mix)`` (deepseek_v4.py:2307/2316).
            # ``residual`` is the state after *this* layer's attention post, and
            # ``post_mix``/``res_mix`` are the *ffn* post/comb this layer computed; the ffn post
            # itself is deferred to the next layer's ``_mhc`` (which is why ``_mhc_post_ffn``
            # takes the ffn params and its outputs are carried across the layer boundary). So
            # the state *leaving* layer i -- the object the reference's ``regions_out[i]`` /
            # ``regions_in[i+1]`` holds -- is reconstructed as
            #     comb @ residual + post_mix * x
            # from this tuple, entirely from engine quantities. Capturing the raw pieces keeps
            # the arithmetic auditable in the consumer instead of hidden in a hook.
            if isinstance(output, (tuple, list)) and len(output) >= 4:
                xo, res, pm, rm = output[0], output[1], output[2], output[3]
                if all(torch.is_tensor(z) for z in (xo, res, pm, rm)) and res.dim() == 3:
                    if (not want_len or int(res.shape[0]) >= want_len) \
                            and int(res.shape[0]) >= _CAP["best_T"]:
                        # .cpu() FIRST, then any cast: casting on the device leaves a
                        # device temporary per layer, and with 43 layers x 4 tensors the
                        # caching allocator holds enough to fail the next 1 GB graph
                        # workspace allocation (observed: PT_DEVMEM Allocation failed for
                        # size::1024MB at the first launch after the hooks engage).
                        _CAP["layer_state"][idx] = {
                            "x": xo.detach().cpu(),
                            "residual": res.detach().cpu(),
                            "post_mix": pm.detach().cpu().float(),
                            "res_mix": rm.detach().cpu().float(),
                        }
            if not torch.is_tensor(t):
                return
            T = int(t.shape[0])
            # The engine pads a prefill to its bucket, so accept any T >= the requested
            # length; the consumer compares the first `want_len` rows (the real tokens).
            if want_len and T < want_len:
                return
            if T < _CAP["best_T"]:
                return
            _CAP["best_T"] = T
            _CAP["regions_out"][idx] = t.detach().to(torch.bfloat16).cpu().clone()
            if len(_CAP["regions_out"]) == len(_layers_ref[0]):
                _write()      # write as soon as the whole chain is present: the engine
        return hook           # workers are SIGTERM'd on shutdown, so atexit may not run

    _layers_ref[0] = layers
    for i, layer in enumerate(layers):
        layer.register_forward_hook(make_hook(i))
    print(f"[region_capture] installed on {len(layers)} decoder-layer regions "
          f"-> {out_dir} (want_len={want_len or 'any'})", flush=True)

    atexit.register(_write)
    return _CAP


def _write():
    """Write the capture now. Called as soon as the chain is complete, and again at exit."""
    if not _CAP["regions_out"] or not _CAP["out"]:
        return
    # The worker process does not export RANK/LOCAL_RANK, so every rank used to write the
    # same filename (last writer wins). Ask the process group instead.
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK"))
    if rank is None:
        try:
            import torch.distributed as dist
            rank = str(dist.get_rank()) if dist.is_initialized() else "0"
        except Exception:  # noqa: BLE001
            rank = "0"
    path = os.path.join(_CAP["out"], f"capture_rank{rank}.pt")
    try:
        torch.save({"L": _CAP["best_T"], "regions": _CAP["regions_out"],
                    "regions_in": _CAP["regions_in"], "regions_out": _CAP["regions_out"],
                    "streams_in": _CAP["streams_in"],
                    "layer_state": _CAP["layer_state"],
                    "ids": _CAP["ids"], "label": _CAP["label"]}, path)
        print(f"[region_capture] wrote {path}: L={_CAP['best_T']} "
              f"in={len(_CAP['regions_in'])} out={len(_CAP['regions_out'])}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[region_capture] write failed: {type(e).__name__}: {e}", flush=True)


def set_ids(ids):
    """Record the token ids the capture corresponds to (called by the driver)."""
    _CAP["ids"] = torch.as_tensor(ids, dtype=torch.long).cpu()
