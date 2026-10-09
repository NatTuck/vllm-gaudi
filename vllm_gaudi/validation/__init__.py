"""Validation-only helpers for the correctness mission (mission step 3).

Nothing here is on the serving path: `region_capture` is installed only when
`DSV4_REGION_CAPTURE` is set, and a capture run is a **validation** run, not a perf run.
"""
