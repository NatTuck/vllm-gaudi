# SPDX-License-Identifier: Apache-2.0
"""LOAD-TIME kernel-variant selection for the fp4_dequant TPC op (DeepSeek V4).

WHAT THIS SELECTS
-----------------
``src/custom_ops/fp4_dequant`` builds ONE TPC kernel, ``fp4_dequant_fwd_gaudi2``,
which decodes packed MXFP4 nibbles to fp8 while folding the per-32 e8m0 scale
exponent ``k`` into the output byte's exponent field (``t = exp + k``).  Two
independent numerical properties of that fold depend on the CHECKPOINT, not on
the input:

  * **the upper guard.**  ``exp in [6, 9]`` for a non-zero nibble, so the guard
    (which saturates the magnitude to ``0x77``) can fire iff
    ``max(k) >= 15 - 9 = 6``, i.e. **iff ``max(scale) > 2**5``**.  For
    ``max(scale) <= 2**5`` the guard provably cannot fire and removing it is
    bit-exact -- MEASURED over the op's whole input space, first divergence at
    ``k = 6``, so ``2**5`` is tight.  Removing it saves 4 instructions/iteration.

  * **the subnormal floor fix.**  The fold's subnormal branch clamps its two
    terms to >= 1 *separately* where their sum should be rounded, so for
    ``t <= -3`` the magnitude floors instead of rounding: finite,
    sign-preserving, silent, unbounded relative error.  The fix is live iff
    ``min(scale) < 2**-8``.

The variant is selected by the UPPER end only, because the upper guard does not
touch the subnormal branch: the two ends are independent, and gating the variant
on the bottom end would select a guarded kernel on this checkpoint, pay the
guard's full cost, and fix nothing there.  The floor fix is therefore present in
BOTH arms, and ``variant`` names ``(guard|noguard) + (|_floor)``.  On this
checkpoint (routed-expert scale bytes ``[118, 126]``) the decision is
``noguard_floor``, admitted, with a 64x margin at the top.

The decision function is the sibling design's (``scratch/clamp77_conditional/
select_variant.py``, device-verified by ``gate_conditional_guard.py``); this
module is the WIRING, and ``src/gates/step_fp4_variant_selection_gate.py``
asserts that the mapping implemented here is the sibling's, over a table of
synthetic ranges including both boundaries.

HOW A VARIANT IS SELECTED IN A PROCESS -- MEASURED
--------------------------------------------------
The four arms are four separate kernel libraries, each binding the PRODUCTION
GUID ``fp4_dequant_fwd_gaudi2`` (one arm per library, so the project's own gates
measure a variant through the production op with no test-side special-casing).
The GUID -> library resolution is the graph compiler's, driven by the
``GC_KERNEL_PATH`` environment variable, which the PT bridge validates as a
**single regular file**.  So variant selection == changing ``GC_KERNEL_PATH``.

Measured on this box (2026-10-12, one card, ``HLS_MODULE_ID=3``, no vLLM;
``scratch/fp4-variant-selection-wiring/probe_gkp_timing.py``, arms A-D, and
``/proc/self/maps``):

  * ``GC_KERNEL_PATH`` is **not** read at process start.  A value set from Python
    *after* ``import habana_frameworks.torch`` and *before* the first HPU graph
    compile is honoured (arm C: output followed the late value).
  * It is read **once, at the first graph compile**, and cached: setting it after
    an unrelated HPU graph has compiled has **no effect** (arm D: the earlier
    library stayed in use).
  * The library actually in use is visible in ``/proc/self/maps`` -- before the
    compile there is no kernel library mapped, after it the one the compiler
    dlopen'd is (arm F2 sees the loss *before* the fp4 op runs).

That last point is what makes this wiring honest rather than hopeful:
``apply_selection`` refuses to *claim* a selection it cannot deliver.  If a
kernel library is already mapped when the load path runs, the process has already
fixed its kernel database and the environment is what is in effect; the wiring
reports that (loudly, naming both libraries) instead of writing an environment
variable that nothing will read.  ``verify_effective()`` closes the loop after
the first compile.

THE ORDERING, AND THE LATE WRITE THAT WAS THE BLOCKER
-----------------------------------------------------
The load path cannot install a kernel library: it runs ~40 s after the worker's
first HPU op, and `GC_KERNEL_PATH` is read at that op.  With the variable at the
image default the wiring used to report ``applied``, write a path nothing would
read, and the first fp4 compile then died with `synStatus 26` -- a promotion that
appeared to work and failed.  Two things changed:

  * the write is **no longer attempted from the load path when the promotion is
    in effect** -- ``apply_selection`` reports ``promoted`` and writes nothing;
  * the promotion is installed by ``install_promoted_kernel_path()``, called from
    ``vllm_gaudi.register_utils()`` -- the ``vllm.general_plugins`` hook vLLM
    invokes in every process at ``WorkerBase.init_worker``, *before* ``init_device``
    and before any ``torch.compile`` (``src/vllm/vllm/v1/worker/worker_base.py``
    :245-247) -- so the ordering is structural.  ``DSV4_FP4_LUT=1`` is what asks
    for it; unset, the hook is a no-op.

  * and the ``too_late`` detector is no longer blind: ``kernel_db_fixed()`` also
    sees the IMAGE DEFAULT library in ``/proc/self/maps``.  Excluding it (as
    ``kernel_libs_mapped()`` still does, for the arm-library case) is exactly what
    made the mis-report possible.

FALLBACK, AND WHY IT IS THE SHIPPED KERNEL
------------------------------------------
When the selected variant's library is not on disk the wiring points
``GC_KERNEL_PATH`` at the **shipped** ``libfp4_dequant_kernels.so`` and says so,
loudly, with the numerical consequence.  That fallback is correct because the
shipped kernel is the *guarded* one: it is the safe top end for every checkpoint
the existing upper-bound assertion admits, and at the bottom end it is the
pre-existing ADMITTED state (the lower-bound guard admits byte 118, whose only
deviations are exact ties).  A fallback therefore never produces a result outside
the range the load-time guards accept.  What it costs is the 4 instructions/iter
of the guard and the floor fix's live class (1,399 weights on this checkpoint) --
both of which are the status quo before this wiring existed, and neither of which
is a new defect.  A *silent* fallback is what this module forbids: the state is
reported on every load, and ``DSV4_FP4_VARIANT_STRICT=1`` turns it into a raise.

ENVIRONMENT
-----------
  DSV4_FP4_LUT              ``1`` -> install the PROMOTED superset library
                            (``libfp4_dequant_lut_kernels.so``) as
                            ``GC_KERNEL_PATH`` at plugin load, i.e. BEFORE the
                            first HPU op.  Unset: no-op.
  DSV4_FP4_VARIANT          ``auto`` (default) | ``off`` | a variant name
                            (``guard``/``noguard``/``guard_floor``/
                            ``noguard_floor``) to force one.
  DSV4_FP4_VARIANT_LIB_DIR  colon-separated directories to search for the arm
                            libraries (default: the promotion target
                            ``src/custom_ops/fp4_dequant/variants/`` and, so the
                            wiring can be exercised before promotion, the
                            reviewed staging directory
                            ``scratch/clamp77_conditional/``).
  DSV4_FP4_VARIANT_STRICT   ``1`` -> raise instead of warn when the selected
                            variant cannot be put into effect.

NOT DONE HERE
-------------
  * **No arm is promoted.**  ``libfp4_dequant_kernels.so`` is never written,
    moved or overwritten; it is the ``GC_KERNEL_PATH`` target of every other run
    in this project and promotion is a separate, reviewed step.
  * **The arms do not yet carry the E3a compressor GUID.**  The shipped library
    exposes TWO GUIDs (``fp4_dequant_fwd_gaudi2`` + ``dsv4_comp_ident``, the E3a
    stub); each arm library exposes ONE (``our_arm_count() == 1`` in its
    ``glue_prod.cpp``).  So pointing ``GC_KERNEL_PATH`` at an arm *removes* the
    E3a GUID from the process.  That is inert unless ``DSV4_COMPRESSOR_OP=1``,
    which is off by default -- and the wiring warns when the combination is
    actually configured.  **It is a blocker for promotion, not for this wiring:**
    the promoted library must carry both GUIDs.
"""
from __future__ import annotations

import ctypes
import dataclasses
import os
import struct

# ---------------------------------------------------------------------------
# the kernel's exponent table, read off the TPC source
# (src/custom_ops/fp4_dequant/kernels/gaudi2/fp4_dequant_fwd_gaudi2.c):
#     exp = idx >= 1 ? 6 + (idx >> 1) : 0     idx = nibble & 7
#     t   = exp + k
#     over = (t_raw >= 15) && nz              -> saturate the magnitude to 0x77
# ---------------------------------------------------------------------------
FP4_EXP_BY_IDX = {0: 0, 1: 6, 2: 7, 3: 7, 4: 8, 5: 8, 6: 9, 7: 9}
GUARD_FIRES_AT_T = 15
E8M0_BIAS = 127

# the largest exponent field a NON-ZERO nibble can carry -- the cheapest case for
# the guard to fire, hence the binding one
MAX_EXP = max(FP4_EXP_BY_IDX[i] for i in range(1, 8))          # 9
MIN_EXP = min(FP4_EXP_BY_IDX[i] for i in range(1, 8))          # 6
K_SAFE_MAX = GUARD_FIRES_AT_T - 1 - MAX_EXP                    # 5
MAX_SCALE_SAFE = 2.0 ** K_SAFE_MAX                             # 32.0
MAX_BYTE_SAFE = K_SAFE_MAX + E8M0_BIAS                         # 132

# the bottom end: the fold is exact for t >= -2, i.e. k >= -8, i.e. byte >= 119.
T_EXACT_MIN = -2
K_EXACT_MIN = T_EXACT_MIN - MIN_EXP                            # -8
MIN_SCALE_EXACT = 2.0 ** K_EXACT_MIN                           # 2**-8
MIN_BYTE_EXACT = K_EXACT_MIN + E8M0_BIAS                       # 119
# the ADMISSION threshold (the load-time lower-bound guard's line): at byte 118
# the only engaged cells are exact TIES, so it is admitted; byte <= 117 admits
# unambiguous over-shoots.  This checkpoint's minimum IS 118 -- zero margin.
MIN_BYTE_ADMIT = 118

# The shipped libfp4_dequant_kernels.so IS the guarded, un-floor-fixed kernel:
# its kernel source is byte-identical (comment-stripped token stream, code_md5
# 5ad02077...) to the sibling's `guard` arm.  So selecting `guard` must use the
# SHIPPED library, not the arm -- identical code, and the shipped one is the
# reviewed artifact (it also carries the E3a GUID).
SHIPPED_VARIANT = "guard"

VARIANTS = ("guard", "noguard", "guard_floor", "noguard_floor")

# ---------------------------------------------------------------------------
# THE PROMOTED LIBRARY, and the ONE thing that decides whether it can be used
# ---------------------------------------------------------------------------
# `libfp4_dequant_lut_kernels.so` is a SUPERSET of the shipped library (it
# carries `fp4_dequant_fwd_gaudi2`, `dsv4_comp_ident_gaudi2` AND the promoted
# `fp4_dequant_lutptr_rc_gaudi2`), so it can be `GC_KERNEL_PATH` for a whole run
# and the LUT path needs no library swap at run time -- the VARIANT travels in
# the 240-byte table, which is runtime data.
#
# THE ORDERING IS THE WHOLE PROBLEM.  `GC_KERNEL_PATH` is read when the graph
# compiler builds its kernel database, at the process's FIRST HPU OP; a write
# after that is never read (MEASURED: Archive/by-date/2026-10-12/fp4-e2e-variant/
# REPORT.md §1/§6, and re-measured by `scratch/fp4-code-fixes/
# accept_promotion_early.py`, which shows the image default appearing in
# /proc/self/maps only AT the first HPU op).  The load path runs ~40 s later, so
# the promotion is installed by `install_promoted_kernel_path()`, which is called
# from `vllm_gaudi.register_utils()` -- the `vllm.general_plugins` hook vLLM
# calls in every process at `WorkerBase.init_worker`, BEFORE `init_device` and
# before any torch.compile (src/vllm/vllm/v1/worker/worker_base.py:245-247).
PROMOTED_LUT_LIB_NAME = "libfp4_dequant_lut_kernels.so"

# ---------------------------------------------------------------------------
# process state (one selection per process; the kernel database is per-process)
# ---------------------------------------------------------------------------
_PREFIX = "[v4-fp4-variant]"

STATE_DISABLED = "disabled"                # DSV4_FP4_VARIANT=off
STATE_NO_DATA = "no_data"                  # no usable scales -> nothing selected
STATE_APPLIED = "applied"                  # the selected library will be / is in use
STATE_FALLBACK_SHIPPED = "fallback_shipped"  # variant absent -> shipped, loudly
STATE_TOO_LATE = "too_late"                # kernel DB already fixed; env in effect
STATE_FOREIGN = "foreign"                  # env names a kernel DB that is not ours

# promotion states (`install_promoted_kernel_path`)
STATE_PROMOTED = "promoted"                # GC_KERNEL_PATH IS the promoted LUT lib
STATE_PROMOTION_OFF = "promotion_off"      # DSV4_FP4_LUT not requested
STATE_PROMOTION_LIB_ABSENT = "promotion_lib_absent"    # promoted lib not on disk
STATE_PROMOTION_FOREIGN = "promotion_foreign"          # a deliberate other lib

# The image default (set by `vendor/habanalabs-env.sh` / `/etc/profile.d/`): not
# one of our libraries, but not a deliberate choice either, so it IS overridable.
_IMAGE_DEFAULT_KERNEL_LIB = "/usr/lib/habanalabs/libtpc_kernels.so"

_STATE: dict = {}


def reset_state() -> None:
    """Clear the recorded selection (tests / a second model in one process)."""
    _STATE.clear()


def selection_state() -> dict:
    return dict(_STATE)


def _warn(msg: str) -> None:
    print(f"{_PREFIX} WARNING: {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"{_PREFIX} {msg}", flush=True)


def _strict() -> bool:
    return os.environ.get("DSV4_FP4_VARIANT_STRICT", "0") == "1"


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Decision:
    """The load-time decision. ``variant`` is None when nothing can be selected."""

    max_scale: float
    min_scale: float
    max_k: int
    min_k: int
    max_byte: int
    min_byte: int
    top_guard_needed: bool      # max byte >  132 -- SELECTS the variant
    floor_fix_needed: bool      # min byte <  119 -- the OTHER condition
    admitted: bool              # both ends inside the window the guards accept
    variant: str | None
    alarms: list

    def report(self) -> str:
        if self.variant is None:
            return ("  no usable scale bounds -> NO variant selected"
                    + "".join(f"\n  ALARM: {a}" for a in self.alarms))
        L = [
            f"  scale range        : [{self.min_scale:g}, {self.max_scale:g}]",
            f"  exponent byte range: [{self.min_byte}, {self.max_byte}]  "
            f"-> k in [{self.min_k}, {self.max_k}]",
            f"  TOP    max(scale) <= 2**5 ({MAX_SCALE_SAFE:g}, byte "
            f"{MAX_BYTE_SAFE})?  {not self.top_guard_needed}   -> the UPPER guard "
            f"can fire: {self.top_guard_needed}",
            f"         (max t = {MAX_EXP} + {self.max_k} = "
            f"{MAX_EXP + self.max_k}, fires at t >= {GUARD_FIRES_AT_T})",
            f"  BOTTOM min(scale) >= 2**-8 ({MIN_SCALE_EXACT:g}, byte "
            f"{MIN_BYTE_EXACT}) [fold exact]?  {not self.floor_fix_needed}",
            f"  ADMIT  byte >= {MIN_BYTE_ADMIT} and byte <= {MAX_BYTE_SAFE}?  "
            f"{self.admitted}",
            f"  => VARIANT: {self.variant.upper()}",
        ]
        L.extend(f"  ALARM: {a}" for a in self.alarms)
        return "\n".join(L)


def e8m0_byte(value: float) -> int:
    """The e8m0 exponent byte for a scale VALUE -- exactly what the kernel sees.

    The op is called with the fp32 scale tensor, and the forward derives the
    fold's ``k`` from its *fp32 exponent field*
    (``vllm_gaudi.extension.ops.decode_e8m0_exponent``:
    ``((w.float().view(int32) >> 23) & 0xFF) - 127``).  So the byte that matters
    is the exponent field, not ``log2`` of a float: for a decoded e8m0 scale
    (an exact power of two -- which is what the load path stores) the two agree
    exactly, and for anything else the exponent field is the floor, which is what
    the kernel folds.  Deriving it here the same way is what makes the predicate
    provably about the kernel's own ``t``.
    """
    try:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    except OverflowError:
        # a value beyond fp32 range cannot be a decoded e8m0 scale; saturate to
        # the top byte so the caller sees "the guard is required" rather than an
        # exception from a diagnostic.
        return 0xFF
    return (bits >> 23) & 0xFF


def _k_of_byte(b: int) -> int:
    return b - E8M0_BIAS


def decide_from_bounds(max_scale: float, min_scale: float) -> Decision:
    """The decision from bounds the load path has ALREADY computed.

    This is the load path's entry point: the upper-bound assertion already
    computes ``max(scale)`` and ``_v4_fp4_scale_lower_bound`` already computes
    ``min(scale)`` (and returns it), so the selection consumes those two numbers
    and needs **no additional pass** over the scale tensors.

    ``min_scale`` may be ``None`` (what ``_v4_fp4_scale_lower_bound`` returns
    when there was nothing to check): then nothing is selected and an alarm is
    reported -- the wiring leaves the environment alone rather than guessing.
    """
    alarms: list = []
    if max_scale is None or min_scale is None:
        return Decision(0.0, 0.0, 0, 0, 0, 255, False, True, False, None,
                        ["no usable routed-expert scale bounds "
                         "(nothing was scanned) -> no variant selected"])
    max_scale = float(max_scale)
    min_scale = float(min_scale)
    if not (max_scale > 0.0) or not (min_scale > 0.0):
        return Decision(max_scale, min_scale, 0, 0, 0, 255, False, True, False,
                        None,
                        [f"non-positive scale bound (max {max_scale!r}, min "
                         f"{min_scale!r}): an e8m0 scale decodes to "
                         f"2**(b-127) > 0, so this is not a decoded e8m0 scale "
                         f"and the byte predicates would be meaningless -> no "
                         f"variant selected"])
    max_byte = e8m0_byte(max_scale)
    min_byte = e8m0_byte(min_scale)

    top_guard_needed = max_byte > MAX_BYTE_SAFE
    floor_fix_needed = min_byte < MIN_BYTE_EXACT
    admitted = (min_byte >= MIN_BYTE_ADMIT) and (max_byte <= MAX_BYTE_SAFE)

    if not admitted:
        if min_byte < MIN_BYTE_ADMIT:
            alarms.append(
                f"min scale byte {min_byte} < {MIN_BYTE_ADMIT}: the fold's "
                f"subnormal branch FLOORS below k = {K_EXACT_MIN} with an "
                f"UNAMBIGUOUS over-shoot (|err| up to 2**-8, finite, "
                f"sign-preserving). The floor fix does not reach this -- the "
                f"lower-bound guard should have raised.")
        if max_byte > MAX_BYTE_SAFE:
            alarms.append(
                f"max scale byte {max_byte} > {MAX_BYTE_SAFE}: the upper guard "
                f"CAN fire; the guarded variant is required -- the upper-bound "
                f"assertion should have raised.")
    elif floor_fix_needed:
        alarms.append(
            f"min scale byte {min_byte} < {MIN_BYTE_EXACT}: the floor fix is "
            f"ACTIVE (only exact ties at byte {MIN_BYTE_ADMIT}, so it is "
            f"admitted).")

    variant = ("guard" if top_guard_needed else "noguard") + \
              ("_floor" if floor_fix_needed else "")
    return Decision(max_scale, min_scale, _k_of_byte(max_byte), _k_of_byte(min_byte),
                    max_byte, min_byte, top_guard_needed, floor_fix_needed,
                    admitted, variant, alarms)


def decide(scale_tensors) -> Decision:
    """The decision from the scale tensors themselves (tests, and callers that
    have the tensors rather than the bounds).

    Accepts the two e8m0 *storage* representations (``uint8`` bytes, or
    ``float8_e8m0fnu``) and already-decoded fp32 values -- the load path stores
    the decoded fp32 form in ``packed_*_scale``.  One pass for max and one for
    min, i.e. the same cost as the assertion this replaces.
    """
    mx, mn, mbyte, nbyte = _bounds(scale_tensors)
    if mbyte is None:
        return decide_from_bounds(mx, mn)
    # byte-native input: build the same Decision from the bytes directly, so a
    # caller holding the raw checkpoint bytes gets byte-exact bounds rather than
    # a round trip through float.
    d = decide_from_bounds(_scale_of_byte(mbyte), _scale_of_byte(nbyte))
    # keep the byte bounds the caller's data actually has (identical for every
    # byte the format defines, but say it rather than assume it)
    assert (d.max_byte, d.min_byte) == (mbyte, nbyte), (d.max_byte, mbyte,
                                                       d.min_byte, nbyte)
    return d


def _scale_of_byte(b: int) -> float:
    """2**(b-127) as the fp32 value the load path would store for byte ``b``."""
    return float(struct.unpack("<f", struct.pack("<I", (b & 0xFF) << 23))[0]) \
        if b != 0 else 2.0 ** -127


def _bounds(scale_tensors):
    """``(max_scale, min_scale, max_byte, min_byte)`` over the scale tensors.

    ``max_byte``/``min_byte`` are ``None`` when only float tensors were seen --
    the caller then derives them from the values.  FAIL CLOSED on the e8m0 NaN
    code point (0xFF) rather than skipping it: the decoders refuse it too
    (``decode_e8m0_scale`` / ``_e8m0_guard``), and silently dropping it here
    would compute a bound from the surviving elements.
    """
    import torch

    mx, mn = 0.0, float("inf")
    mbyte, nbyte = None, None
    for t in scale_tensors:
        if t is None or t.numel() == 0:
            continue
        if t.dtype == torch.uint8 or t.dtype == torch.float8_e8m0fnu:
            b = t.view(torch.uint8) if t.dtype == torch.float8_e8m0fnu else t
            if int(b.max()) == 0xFF:
                raise ValueError(
                    "fp4 variant selection: e8m0 scale byte 0xFF (reserved for "
                    "NaN, OCP MX 5.4.1) present; the checkpoint is corrupt -- "
                    "see decode_e8m0_scale / _e8m0_guard. Refusing to compute a "
                    "bound.")
            bmax, bmin = int(b.max()), int(b.min())
            mbyte = bmax if mbyte is None else max(mbyte, bmax)
            nbyte = bmin if nbyte is None else min(nbyte, bmin)
            mx = max(mx, _scale_of_byte(bmax))
            mn = min(mn, _scale_of_byte(bmin))
        else:
            a = t.float().abs()
            if a.numel():
                mx = max(mx, float(a.max()))
                mn = min(mn, float(a.min()))
    if mn == float("inf"):
        mn = 0.0
    return mx, mn, mbyte, nbyte


# ---------------------------------------------------------------------------
# resolving a variant to a kernel library
# ---------------------------------------------------------------------------
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _shipped_lib() -> str:
    return os.path.join(_REPO, "src", "custom_ops", "fp4_dequant",
                        "libfp4_dequant_kernels.so")


def promoted_lut_lib() -> str:
    """The promoted SUPERSET library: shipped kernels + the LUT kernel."""
    return os.path.join(_REPO, "src", "custom_ops", "fp4_dequant",
                        PROMOTED_LUT_LIB_NAME)


def promotion_requested() -> bool:
    """`DSV4_FP4_LUT=1`: this run wants the promoted LUT path."""
    return os.environ.get("DSV4_FP4_LUT", "0").strip() == "1"


def promotion_in_effect() -> bool:
    """Is `GC_KERNEL_PATH` the promoted LUT library (and does it exist)?"""
    cur = os.environ.get("GC_KERNEL_PATH", "")
    if not cur or not os.path.isfile(cur):
        return False
    return _real(cur) == _real(promoted_lut_lib())


def _kernel_libs_in_maps() -> list:
    """Every kernel-database library this process has mapped, image default too.

    Deliberately NOT filtered against ``known_kernel_libs()``: the image default
    is the one that matters when the promotion was never installed, and leaving
    it out is what made the wiring blind to "the database is already fixed".
    ``libtpcOpsValidation.so`` is mapped at habana init, before any op, and is
    NOT a kernel database -- it is excluded.
    """
    want = {_real(_IMAGE_DEFAULT_KERNEL_LIB), _real(promoted_lut_lib())}
    cur = os.environ.get("GC_KERNEL_PATH", "")
    if cur:
        want.add(_real(cur))
    out = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "/" not in line:
                    continue
                p = line.split()[-1]
                if "libtpc_kernels" in os.path.basename(p) \
                        or "libfp4_dequant" in os.path.basename(p):
                    if p not in out:
                        out.append(p)
    except OSError:
        return []
    # keep the library paths this wiring cares about first (the one in
    # `GC_KERNEL_PATH`, then the image default / the promoted library), so a
    # caller can name the database actually in use; anything else follows.
    return sorted(out, key=lambda p: (0 if _real(p) in want else 1, p))


def kernel_db_fixed() -> bool:
    """Has an HPU op already run in this process, fixing the kernel database?

    The signal is `/proc/self/maps`: with `GC_KERNEL_PATH` at the image default
    the default library is mapped *at the first HPU op* and not before --
    MEASURED in `scratch/fp4-code-fixes/accept_promotion_early.py`
    (`libtpc_kernels.so` absent after import, present after one op).  The graph
    compiler reads `GC_KERNEL_PATH` when it builds that database, so a library
    mapped there means a later write cannot take effect in THIS process.

    This is a *process* signal, not a proof about a child process: an env write
    made after the parent's first op is still inherited by workers spawned
    later, which is why `install_promoted_kernel_path` warns on it but does not
    refuse.  Returns False when /proc cannot be read (a "could not consult",
    which must not read as "not fixed"): the caller then simply proceeds, as
    before.
    """
    return bool(_kernel_libs_in_maps())


def search_dirs() -> list:
    """Directories searched for the arm libraries, in order.

    ``src/custom_ops/fp4_dequant/variants/`` is the promotion target: it is
    searched first so promotion needs no code change.  The staging directory the
    reviewed arms live in today is searched second, so the wiring can be
    exercised end-to-end before promotion; it should be dropped once the arms are
    promoted (and is dropped automatically if it does not exist).
    """
    out = []
    env = os.environ.get("DSV4_FP4_VARIANT_LIB_DIR")
    if env:
        out.extend(p for p in env.split(os.pathsep) if p)
    out.append(os.path.join(_REPO, "src", "custom_ops", "fp4_dequant", "variants"))
    out.append(os.path.join(_REPO, "scratch", "clamp77_conditional"))
    return out


def variant_lib_candidates(variant: str) -> list:
    """Every filename a given variant's library may have, per search dir."""
    names = (f"libfp4_dequant_arm_{variant}.so", f"libfp4_cond_{variant}.so")
    return [os.path.join(d, n) for d in search_dirs() for n in names]


def resolve_variant_lib(variant: str):
    """The library implementing ``variant``, or None.

    The shipped library IS the ``guard`` variant, so ``guard`` resolves to it --
    identical code, and it is the reviewed artifact (and the only one carrying
    the E3a GUID).
    """
    if variant == SHIPPED_VARIANT:
        p = _shipped_lib()
        return p if os.path.isfile(p) else None
    for p in variant_lib_candidates(variant):
        if os.path.isfile(p):
            return p
    return None


def _real(p: str) -> str:
    return os.path.realpath(p)


def _lib_label(path: str) -> str:
    """A human label for a kernel library the wiring did not choose itself."""
    known = known_kernel_libs()
    label = known.get(_real(path))
    if label is not None:
        return label
    if _real(path) == _real(_IMAGE_DEFAULT_KERNEL_LIB):
        return "image_default"
    return "unknown"


def known_kernel_libs() -> dict:
    """realpath -> label, for every library this wiring can put into effect.

    Every ARM library is listed too, including the ``guard`` arm: a run whose
    environment points ``GC_KERNEL_PATH`` straight at an arm (which is how the
    project's own gates work) must still be *detectable* as "a kernel library is
    already mapped", or the wiring would believe it can still select.  The
    PROMOTED LUT library is listed for the same reason: it is ours, so the
    wiring must recognise it (and must not treat it as a foreign database it
    has no business overriding).
    """
    out = {}
    shipped = _shipped_lib()
    if os.path.isfile(shipped):
        out[_real(shipped)] = "shipped(=guard)"
    promoted = promoted_lut_lib()
    if os.path.isfile(promoted):
        out[_real(promoted)] = "promoted_lut"
    for v in VARIANTS:
        for p in variant_lib_candidates(v):
            if os.path.isfile(p):
                out.setdefault(_real(p), f"{v}_arm" if v == SHIPPED_VARIANT else v)
    return out


def kernel_libs_mapped() -> list:
    """Which of the KNOWN kernel libraries the process has already dlopen'd.

    The graph compiler caches its kernel database at the first compile, so a
    library that is already mapped means the process's kernel is already fixed
    and a later ``GC_KERNEL_PATH`` write cannot change it (MEASURED: probe arm
    D/F2).  Filtered against ``known_kernel_libs()`` so the op *extension*
    (``hpu_custom_fp4_dequant*.so``) is not mistaken for a kernel library.
    """
    known = known_kernel_libs()
    out = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "/" not in line:
                    continue
                p = line.split()[-1]
                if p in known and p not in out:
                    out.append(p)
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# applying the selection
# ---------------------------------------------------------------------------
def _mode() -> str:
    return os.environ.get("DSV4_FP4_VARIANT", "auto").strip().lower()


def _record(state: str, **kw) -> str:
    _STATE.clear()
    _STATE.update(state=state, **kw)
    return state


def install_promoted_kernel_path(where: str = "process start") -> str:
    """Point `GC_KERNEL_PATH` at the PROMOTED LUT library -- BEFORE the first HPU op.

    THIS IS THE FIX FOR THE LATE WRITE.  The load path's write at the end of
    `apply_selection` happens after the first HPU op, so with `GC_KERNEL_PATH` at
    the image default it reported ``applied`` and then the first fp4 compile
    died with `synStatus 26` (MEASURED: `Archive/by-date/2026-10-12/
    fp4-e2e-variant/REPORT.md` §6A, re-produced by
    `scratch/fp4-code-fixes/accept_promotion_early.py`).  Promotion sidesteps the
    write entirely: the superset library carries the LUT kernel, so it can be the
    kernel database for the whole run and the variant travels in the table.

    Called from `vllm_gaudi.register_utils()` (the `vllm.general_plugins` hook,
    before `init_device` in every process) and importable on its own, so the
    ordering is structural rather than hoped for.  Idempotent; it writes nothing
    unless `DSV4_FP4_LUT=1` asks for the promotion.

    WHAT MUST PRECEDE THE WRITE, all checked here:
      1. the promotion is requested (`DSV4_FP4_LUT=1`);
      2. the promoted library exists -- a missing one would point
         `GC_KERNEL_PATH` at nothing and fail EVERY compile;
      3. no deliberate operator choice is clobbered (a non-default library that
         is not one of ours is left alone, loudly) -- the same precedence rule
         `apply_selection` uses;
      4. the ordering itself: if the kernel database is already fixed in THIS
         process the write cannot take effect here, which is reported loudly --
         but the write still happens, because the environment is inherited by
         any worker spawned later and an env write after the parent's first op is
         not late for a child process.
    """
    lib = promoted_lut_lib()
    if not promotion_requested():
        return _record(STATE_PROMOTION_OFF, where=where, lib=lib, variant=None,
                       mapped=kernel_libs_mapped())
    if not os.path.isfile(lib):
        _warn(f"promotion requested (DSV4_FP4_LUT=1) but the promoted library "
              f"{lib!r} is not on disk; GC_KERNEL_PATH="
              f"{os.environ.get('GC_KERNEL_PATH', '')!r} is left exactly as "
              f"configured. Build it with src/custom_ops/fp4_dequant/build_lut.sh.")
        return _record(STATE_PROMOTION_LIB_ABSENT, where=where, lib=lib,
                       variant=None, mapped=kernel_libs_mapped())

    cur = os.environ.get("GC_KERNEL_PATH", "")
    if cur and _real(cur) == _real(lib):
        _info(f"{where}: the promoted LUT library is already GC_KERNEL_PATH "
              f"({lib})")
        return _record(STATE_PROMOTED, where=where, lib=lib, previous=cur,
                       variant=None, mapped=kernel_libs_mapped(), early=True)

    known = known_kernel_libs()
    cur_exists = bool(cur) and os.path.isfile(cur)
    if cur_exists and _real(cur) != _real(_IMAGE_DEFAULT_KERNEL_LIB) \
            and _real(cur) not in known:
        _warn(f"{where}: GC_KERNEL_PATH={cur!r} names a kernel library that is "
              f"not one of this wiring's candidates and is not the image "
              f"default -- treating it as a deliberate operator choice and "
              f"leaving it alone. The promotion ({lib}) was NOT installed. Set "
              f"DSV4_FP4_LUT=0 to silence this, or point GC_KERNEL_PATH at the "
              f"promoted library.")
        return _record(STATE_PROMOTION_FOREIGN, where=where, lib=lib,
                       previous=cur, variant=None, mapped=kernel_libs_mapped())

    late = kernel_db_fixed()
    os.environ["GC_KERNEL_PATH"] = lib
    if os.environ.get("GC_KERNEL_PATH") != lib:
        raise RuntimeError(f"{_PREFIX} failed to set GC_KERNEL_PATH to {lib!r}")
    if late:
        _warn(f"{where}: GC_KERNEL_PATH {cur!r} -> {lib!r}, but this process has "
              f"ALREADY run an HPU op ({_kernel_libs_in_maps()}), so its kernel "
              f"database is fixed and the promotion does NOT take effect HERE. "
              f"The environment is still written because a worker process "
              f"spawned from this one inherits it, and the load path verifies the "
              f"ACTIVE library's GUID table before using the LUT kernel.")
    else:
        _info(f"{where}: PROMOTION installed before the first HPU op -- "
              f"GC_KERNEL_PATH {cur!r} -> {lib!r}")
    return _record(STATE_PROMOTED, where=where, lib=lib, previous=cur,
                   variant=None, mapped=kernel_libs_mapped(), early=not late,
                   db_already_fixed=late)


def apply_selection(decision: Decision, where: str = "load") -> str:
    """Put the decision into effect. Returns the state string (also recorded).

    Called from the load path with the bounds it has already computed.  The
    library in use is per-PROCESS (the graph compiler's kernel database), so this
    is idempotent: a second call sees the library already mapped and reports
    ``applied``.

    WHEN THE PROMOTION IS IN EFFECT THIS FUNCTION WRITES NOTHING.  The promoted
    superset library is already `GC_KERNEL_PATH` (installed before the first HPU
    op by `install_promoted_kernel_path`), it carries the LUT kernel, and the
    VARIANT is delivered as the 240-byte table by the LUT path -- so there is no
    library to select, and writing an arm path here would both be unread and
    UNDO the promotion.  The state reported is ``promoted``, not ``applied``:
    this call did not apply a library, and saying it did is the mis-report this
    wiring exists to prevent.
    """
    mode = _mode()
    if mode == "off":
        _info(f"selection disabled (DSV4_FP4_VARIANT=off) at {where}; "
              f"GC_KERNEL_PATH={os.environ.get('GC_KERNEL_PATH', '')!r} is left "
              f"exactly as configured")
        return _record(STATE_DISABLED, where=where, variant=None,
                       mapped=kernel_libs_mapped())

    variant = decision.variant if decision is not None else None
    forced = False
    if mode not in ("auto", ""):
        if mode not in VARIANTS:
            raise ValueError(
                f"DSV4_FP4_VARIANT={mode!r} is not one of auto/off/{VARIANTS}")
        variant = mode
        forced = True
    if variant is None:
        _warn(f"no variant selected at {where} "
              f"({'/'.join(decision.alarms) if decision else 'no decision'}); "
              f"GC_KERNEL_PATH={os.environ.get('GC_KERNEL_PATH', '')!r} is left "
              f"exactly as configured")
        return _record(STATE_NO_DATA, where=where, variant=None,
                       mapped=kernel_libs_mapped())

    # (0) THE PROMOTION.  The library the compiler will use is the promoted
    #     superset, installed before the first HPU op; the variant is the table.
    if promotion_in_effect():
        _info(f"{where}: the promoted LUT library is GC_KERNEL_PATH "
              f"({promoted_lut_lib()}); the checkpoint selects "
              f"{variant.upper()} and it is delivered as the 240-byte TABLE by "
              f"the LUT path, not by a library swap -- no GC_KERNEL_PATH write "
              f"here.")
        return _record(STATE_PROMOTED, where=where, variant=variant,
                       lib=promoted_lut_lib(), delivered_by="lut_table",
                       mapped=kernel_libs_mapped(), effective=True)

    selected = resolve_variant_lib(variant)
    shipped = _shipped_lib()
    mapped = kernel_libs_mapped()
    known = known_kernel_libs()

    # (1) Has this process already fixed its kernel database?  Then the
    #     environment value is what is in effect and writing it changes nothing.
    #     `kernel_db_fixed()` is the honest test: it also sees the IMAGE DEFAULT,
    #     which `kernel_libs_mapped()` deliberately excludes -- that exclusion is
    #     what made the wiring report `applied` with GC_KERNEL_PATH at the image
    #     default and then die with synStatus 26 (MEASURED, see the module
    #     docstring).  A database that is already fixed can never be changed by a
    #     write here, whatever library it holds.
    db_fixed = kernel_db_fixed()
    if mapped or db_fixed:
        in_use = mapped[0] if mapped else _kernel_libs_in_maps()[0]
        if selected is not None and _real(selected) == in_use:
            _info(f"{where}: the {variant} library is already the one in use "
                  f"({in_use})")
            return _record(STATE_APPLIED, where=where, variant=variant,
                           lib=selected, mapped=mapped, effective=True)
        msg = (f"{where}: the process has ALREADY run an HPU op, so its "
               f"kernel database is fixed and GC_KERNEL_PATH is no longer read "
               f"(MEASURED). The kernel in use is {in_use} "
               f"[{known.get(in_use, _lib_label(in_use))}]; the checkpoint selects "
               f"{variant!r} ({selected or 'library not on disk'}). This is NOT "
               f"applied -- the run continues on the library the environment "
               f"named. Set GC_KERNEL_PATH before the first HPU graph compile to "
               f"select a variant.")
        if _strict():
            raise RuntimeError(f"{_PREFIX} {msg}")
        _warn(msg)
        return _record(STATE_TOO_LATE, where=where, variant=variant,
                       lib=selected, mapped=mapped, effective=False)

    # (2) Nothing compiled yet -> our write will be read at the first compile.
    #     Do not clobber a kernel database the operator chose deliberately: this
    #     wiring owns the choice among ITS candidates, not the choice of an
    #     entirely different library (e.g. an engine run pointed at one of the
    #     other arms under src/custom_ops/fp4_dequant_*/ for an A/B).
    #
    #     THE PATH MUST EXIST to count as a deliberate choice.  A path that is
    #     not there is a MISCONFIGURATION, and treating it as deliberate would
    #     mean the process proceeds with nothing at all -- no selection, no
    #     fallback warning, no error -- which is the exact silent-failure shape
    #     this wiring exists to remove.  A missing path therefore falls through
    #     to the normal logic below (`fallback_shipped`, or `applied`).
    #
    #     PRECEDENCE -- explicit beats implicit.  `foreign` guards the AUTOMATIC
    #     path (DSV4_FP4_VARIANT=auto) against an accidental clobber of another
    #     agent's A/B.  Naming a variant is a strictly more specific instruction
    #     from the person launching THIS run, so it wins -- as the existing `off`
    #     already does -- and it is announced, so it is never a silent clobber.
    cur = os.environ.get("GC_KERNEL_PATH", "")
    cur_exists = bool(cur) and os.path.isfile(cur)
    cur_is_foreign = (cur_exists
                      and _real(cur) != _real(_IMAGE_DEFAULT_KERNEL_LIB)
                      and _real(cur) not in known)
    if cur_is_foreign and not forced:
        msg = (f"{where}: GC_KERNEL_PATH={cur!r} names a kernel library that is "
               f"not one of this wiring's candidates and is not the image "
               f"default -- treating it as a deliberate operator choice and "
               f"leaving it alone. The checkpoint selects {variant!r}"
               + (f" ({selected})" if selected else " (library not on disk)")
               + ". Set DSV4_FP4_VARIANT=off to silence this, name a variant to "
                 "override it, or point GC_KERNEL_PATH at a candidate.")
        if _strict():
            raise RuntimeError(f"{_PREFIX} {msg}")
        _warn(msg)
        return _record(STATE_FOREIGN, where=where, variant=variant,
                       lib=selected, previous=cur, mapped=[], effective=False)
    if cur_is_foreign and forced:
        _warn(f"{where}: explicit DSV4_FP4_VARIANT={mode!r} OVERRIDES the "
              f"deliberately-set GC_KERNEL_PATH={cur!r} (which is not one of this "
              f"wiring's candidates). The explicit instruction wins; say so if "
              f"that was not what you wanted.")
    elif cur and not cur_exists:
        _warn(f"{where}: GC_KERNEL_PATH={cur!r} does not exist -- a "
              f"MISCONFIGURATION, not a deliberate choice, so it is NOT treated "
              f"as foreign; the normal selection/fallback below replaces it.")

    target = selected if selected is not None else shipped
    if target is None or not os.path.isfile(target):
        raise RuntimeError(
            f"{_PREFIX} neither the {variant} variant library nor the shipped "
            f"kernel library exists ({target!r}); searched "
            f"{search_dirs()}. Refusing to continue without a kernel library.")

    fallback = selected is None
    if fallback:
        msg = (f"{where}: the {variant!r} variant library is NOT on disk "
               f"(searched {search_dirs()}); falling back to the SHIPPED kernel "
               f"{shipped}. The shipped kernel is the guarded one, so the "
               f"fallback is safe at the top end for every checkpoint the "
               f"upper-bound assertion admits, and at the bottom end it is the "
               f"admitted (pre-existing) state; what it costs is the guard's 4 "
               f"instructions/iteration"
               + (", and it leaves the FLOOR FIX INACTIVE on this checkpoint, "
                  "which needs it" if decision and decision.floor_fix_needed
                  else "")
               + ". Promotion of the arms is the separate step that removes "
                 "this fallback.")
        if _strict():
            raise RuntimeError(f"{_PREFIX} {msg}")
        _warn(msg)
        if os.environ.get("DSV4_COMPRESSOR_OP", "0") == "1":
            _warn(f"DSV4_COMPRESSOR_OP=1 with a non-shipped kernel library: the "
                  f"arm libraries expose ONE GUID (fp4_dequant_fwd_gaudi2) while "
                  f"the shipped library exposes TWO (+ dsv4_comp_ident, the E3a "
                  f"stub). The E3a node will fail to instantiate. Rebuild the "
                  f"arm with the stub GUID before using it here.")

    prev = os.environ.get("GC_KERNEL_PATH")
    os.environ["GC_KERNEL_PATH"] = target
    if os.environ.get("GC_KERNEL_PATH") != target:
        raise RuntimeError(f"{_PREFIX} failed to set GC_KERNEL_PATH to {target!r}")
    state = STATE_FALLBACK_SHIPPED if fallback else STATE_APPLIED
    _info(f"{where}: checkpoint selects {variant.upper()} -> GC_KERNEL_PATH "
          f"{prev!r} -> {target!r}"
          + (" (FALLBACK: variant library absent)" if fallback else ""))
    return _record(state, where=where, variant=variant, lib=target,
                   previous=prev, mapped=[], effective=True,
                   fallback=fallback)


# ---------------------------------------------------------------------------
# the promoted kernel: is its NAME in the ACTIVE library's GetKernelGuids table?
# ---------------------------------------------------------------------------
DEVICE_ID_GAUDI2 = 3
LUT_KERNEL_NAME = "fp4_dequant_lutptr_rc_gaudi2"


class _GuidInfo(ctypes.Structure):
    """``tpc_lib_api::GuidInfo``; ``name`` is ``char[MAX_NODE_NAME]``, 64."""

    _fields_ = [("name", ctypes.c_char * 64),
                ("nameHash", ctypes.c_uint64),
                ("kernelProperties", ctypes.c_uint32)]


def kernel_lib_guids(path: str, device_id: int = DEVICE_ID_GAUDI2):
    """The GUID names ``path`` serves for ``device_id``, read from the artifact.

    This calls the library's OWN exported ``GetKernelGuids`` -- the exact
    function the graph compiler calls, in the same two-call shape (query the
    count, then fill) -- so it answers the question that decides promotion
    ("is the kernel's NAME in the ACTIVE library's table?") rather than a proxy
    for it.  The table is per-``DeviceId``: the same library answers 2021 /
    3252 / 3358 GUIDs for Gaudi / Gaudi2 / Gaudi3 (MEASURED), which is why the
    device id is a parameter and not a default.

    Returns ``None`` when the file cannot be dlopen'd or does not export
    ``GetKernelGuids`` -- an explicit "could not consult", never an empty list
    that would read as "the table says no".
    """
    import ctypes as _ct

    try:
        lib = _ct.CDLL(path)
        fn = lib.GetKernelGuids
    except (OSError, AttributeError) as exc:
        _warn(f"could not read the GUID table of {path!r}: "
              f"{type(exc).__name__}: {exc}")
        return None
    fn.restype = _ct.c_int
    fn.argtypes = [_ct.c_int, _ct.POINTER(_ct.c_uint32),
                   _ct.POINTER(_GuidInfo)]
    n = _ct.c_uint32(0)
    if fn(device_id, _ct.byref(n), None) != 0:
        return None
    total = n.value
    if total == 0:
        return []
    buf = (_GuidInfo * total)()
    cap = _ct.c_uint32(total)
    if fn(device_id, _ct.byref(cap), buf) != 0:
        return None
    return [bytes(buf[i].name).split(b"\0", 1)[0].decode()
            for i in range(cap.value)]


def lut_kernel_present(path: str) -> bool | None:
    """Is the promoted LUT kernel in the table of the library at ``path``?

    ``None`` means the table could not be consulted (see ``kernel_lib_guids``).
    """
    names = kernel_lib_guids(path)
    if names is None:
        return None
    return LUT_KERNEL_NAME in names


def verify_effective() -> dict:
    """Confirm, after the fact, which kernel library the compiler actually loaded.

    Meaningful only once a graph has compiled.  Returns a dict; ``effective`` is
    None while nothing is mapped yet (the compile has not happened), True when
    the selected library is the one mapped, False when it is not.
    """
    known = known_kernel_libs()
    mapped = kernel_libs_mapped()
    want = _STATE.get("lib")
    if not mapped:
        return dict(effective=None, mapped=[], expected=want,
                    note="no kernel library mapped yet: no graph has compiled")
    in_use = mapped[0]
    ok = want is not None and _real(want) == in_use
    out = dict(effective=ok, mapped=mapped, expected=want,
               in_use=in_use, in_use_label=known.get(in_use, "unknown"))
    if not ok:
        _warn(f"the kernel library in use is {in_use} "
              f"[{out['in_use_label']}] but the selection wanted "
              f"{want!r} [{_STATE.get('variant')}] -- the selection was NOT "
              f"honoured")
    return out


def decide_and_apply_from_bounds(max_scale, min_scale, where="load") -> dict:
    """The load path's one-liner: decide from already-computed bounds, then apply."""
    d = decide_from_bounds(max_scale, min_scale)
    if _mode() != "off":
        _info(f"{where}: {d.report()}")
    state = apply_selection(d, where=where)
    return dict(decision=d, state=state)
