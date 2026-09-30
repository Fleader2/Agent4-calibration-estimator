# Calibration-Target Eligibility Policy (`eligibility-v1`)

Implemented in `app.agent4.eligibility`. This document formalizes exactly which
parameters/initial conditions Agent 4 will calibrate, and why — the single most important
policy in this repository, since it is the concrete mechanism behind the central rule "Agent 4
must never overwrite curated/evidence-backed values."

## Parameter sources (Agent 2's own `ParameterSource` enum)

Confirmed by direct inspection of `agent2-antimony-builder`'s `app/agent2/types.py` this
session, the complete vocabulary is:

```
CURATED, LITERATURE_DERIVED, AI_PREDICTED, DERIVED_FROM_MACRO_KINETICS,
DERIVED_FROM_POOL_CONSERVATION, HEURISTIC_INITIALIZATION, DEFAULT, PLACEHOLDER, CALIBRATED
```

`CALIBRATED` is explicitly reserved, in Agent 2's own code and docstrings, for a value Agent 4
itself produces — Agent 2 never assigns it to anything it produces itself. Agent 2's own
`ParameterSpecification.is_calibrated` property exists specifically to recognize Agent 4's own
output when it is fed back in.

## Default-calibratable sources

```python
DEFAULT_CALIBRATABLE_PARAMETER_SOURCES = {"HEURISTIC_INITIALIZATION", "PLACEHOLDER", "DEFAULT"}
```

A parameter with one of these sources is calibratable **by simply being listed** as a
`CalibrationTarget` — no further authorization is needed. `DEFAULT` is included alongside
`HEURISTIC_INITIALIZATION` because Agent 2's own code (`app/agent2/boundaries/policy.py`) groups
them together for boundary-likelihood purposes; both represent a non-evidence-backed,
default-only value.

## Protected sources

```python
PROTECTED_PARAMETER_SOURCES = {
    "LITERATURE_DERIVED",
    "CURATED",
    "AI_PREDICTED",
    "DERIVED_FROM_MACRO_KINETICS",
    "DERIVED_FROM_POOL_CONSERVATION",
    "CALIBRATED",
}
```

A parameter with one of these sources is calibratable **only** if the `CalibrationRequest`
itself names either the exact source string in `authorized_provenance_classes`, or the exact
parameter id in `authorized_parameter_ids`. Neither of these is inferred or defaulted — the
caller must say so explicitly, every time. `CALIBRATED` is included in the protected set: Version
1 does not treat "already the output of a prior calibration run" as automatically re-fittable;
re-calibrating a calibrated value requires the same explicit authorization as any other
protected source.

An unrecognized `source` string (neither in the default-calibratable nor the protected set) is
always rejected — Agent 4 never guesses which category a source it doesn't recognize belongs to.

## The absolute veto: `fixed=True`

A parameter with `fixed=True` (Agent 2's own declared field) is **never** calibratable, under
any circumstances, regardless of its `source` or any authorization the request supplies. This
is checked first, before the source-based logic, and there is no override anywhere in the
`CalibrationRequest` contract that can unlock it. `app.agent4.pipeline` additionally re-verifies
this as a defensive, literal post-hoc invariant: if a fixed parameter's id is ever found among
the produced `ParameterEstimate`s (which given the eligibility check above should be structurally
impossible), the pipeline raises `Agent4Error` rather than returning a corrupted report.

Empirically, `fixed=True` is not set on any parameter in the current real `sce00061` model (0 of
109) — this rule exists for when it is.

## Initial conditions

Unlike parameters, there is no "default-calibratable" class of initial condition. **Listing a
species as an `INITIAL_CONDITION` target in the request is itself Version 1's own explicit-
authorization mechanism** — there is nothing else required to unlock it, unless Agent 2 also
happened to record a protected `initialization_source` for that species (e.g. `"CURATED"`), in
which case the same provenance-class/parameter-id authorization required for parameters applies,
substituting the species id for a parameter id in `authorized_parameter_ids`.

On the current real `sce00061` model, no species has any declared `initialization_source` (0 of
53) — so today, every species initial condition is calibratable simply by being listed as a
target (and by having explicit bounds supplied, see below).

## Bounds resolution

A target's bounds are resolved in this order, and the target is rejected
(`CalibrationStatus.INVALID_REQUEST`) if neither applies:

1. The request's own `CalibrationTarget.bounds`, if given.
2. For a `PARAMETER` target only: Agent 2's own declared `lower_bound`/`upper_bound`, if **both**
   are set.

Agent 4 never invents a numeric bound. Since the real `sce00061` model currently has zero
parameters with any declared bound, calibrating any real parameter today requires the caller to
supply explicit bounds in the request itself — confirmed and exercised in this repository's own
real-integration test.

## Request-level validation (`app.agent4.eligibility.validate_request`)

Beyond the per-target checks above, an entire request is rejected if:

* `targets` is empty.
* Any two targets name the same `(target_kind, target_id)` pair (a duplicate).
* A target names a parameter or species id that does not exist in the `agent2_model`.
* Any observation names a `target_species_id` that does not exist in the `agent2_model`.

Every violation is collected (not just the first) and reported together in
`CalibrationResult.optimizer_message` when the request is rejected — a caller sees every problem
at once, not one at a time across repeated resubmissions.
