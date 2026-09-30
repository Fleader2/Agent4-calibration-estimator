# Identifiability Diagnostics (`identifiability-v1`)

Implemented in `app.agent4.identifiability`. **None of the checks below is, or claims to be, a
formal structural-identifiability proof** (e.g. differential-algebra-based observability
analysis, or profile-likelihood confidence intervals). Every one is a local, approximate,
practical heuristic evaluated near the fitted optimum or across the already-computed multi-start
results — "modest but useful," per the task's own explicit framing.

## 1. Bound hits (`IdentifiabilityCategory.BOUND_HIT`)

`app.agent4.identifiability.is_at_bound(value, lower, upper)` classifies a fitted value as at its
lower/upper bound when it sits within `1%` of the bound's own range
(`_BOUND_HIT_RELATIVE_TOLERANCE = 0.01`) of that edge. A parameter converging to its own bound
usually means either the bound is too tight, or the data does not actually constrain the
parameter away from that edge — the point estimate is reported, but flagged as unreliable.

**Establishes:** the optimizer's own final answer sits at an artificial edge.
**Does not establish:** what the "correct" unconstrained value would have been.

## 2. Low sensitivity (`IdentifiabilityCategory.LOW_SENSITIVITY`)

For each target, `app.agent4.identifiability.low_sensitivity_findings` nudges that one target by
1% of its own bound range (holding every other target fixed at the fitted optimum) and
re-evaluates the training objective. If the **absolute** change in objective falls below
`max(_LOW_SENSITIVITY_ABSOLUTE_FLOOR, _LOW_SENSITIVITY_RELATIVE_THRESHOLD * baseline_objective)`
(a combined absolute-floor/relative criterion — `1e-8` absolute, `1e-6` relative), the target is
flagged. The absolute floor exists specifically to stay correct even when the baseline objective
is itself near-zero (a noiseless, near-perfectly-fit synthetic case): a purely relative
comparison against a near-zero denominator would otherwise be dominated by the simulation/
interpolation pipeline's own numerical noise floor (empirically observed around `1e-11` for the
small models this Version 1 policy targets) rather than the target's real effect.

**Establishes:** the supplied observations carry little local information about this one
target's value, independent of the others.
**Does not establish:** global insensitivity across the whole parameter space, or insensitivity
in combination with other parameters (see correlation, below).

## 3. High correlation (`IdentifiabilityCategory.HIGH_CORRELATION`)

`app.agent4.identifiability.correlation_findings` builds a central-difference Jacobian of the
full training residual vector with respect to every target at the fitted optimum, forms the
Fisher-information approximation `J^T J`, inverts it (`numpy.linalg.pinv`, robust to
near-singularity) to approximate the parameter covariance matrix, and flags any pair whose
implied correlation coefficient exceeds `0.95` in absolute value. This is the closest Version 1
comes to a "practical identifiability" statement about **pairs** of parameters: a structurally
non-identifiable product (e.g. `k1*k2` appearing together in every rate law, as in this
repository's own deliberately-unidentifiable synthetic fixture) produces a near-singular Fisher
information and a correlation at or extremely close to `±1`.

**Establishes:** a local linearization suggests two targets are practically indistinguishable
given the supplied data, near this particular optimum.
**Does not establish:** a global or structural non-identifiability proof valid everywhere in
parameter space — a different optimum, or additional/different observations, could in principle
break the correlation.

## 4. Degenerate objective across multi-starts (`IdentifiabilityCategory.DEGENERATE_OBJECTIVE`)

`app.agent4.identifiability.degenerate_objective_findings` compares every pair of *converged*
multi-start results: if their final parameter vectors are far apart (normalized Euclidean
distance, by each target's own bound range, exceeding `0.2`) while their objective values are
nearly identical (relative difference below `1e-3`), the pair is flagged — this is the only
`CRITICAL`-severity finding category, and its presence is what elevates a run's overall
`CalibrationStatus` to `UNIDENTIFIABLE` (see `app.agent4.pipeline`).

**Establishes:** the optimizer, starting from genuinely different points, converged to
meaningfully different answers that fit the data almost equally well — the objective surface is
essentially flat across a non-trivial region, so no single fitted value from this run is a
trustworthy unique answer.
**Does not establish:** the full extent or shape of that flat region, or whether it is truly
infinite (structural) versus merely broad (practical, given this particular dataset).

## 5. Insufficient observations (`IdentifiabilityCategory.INSUFFICIENT_OBSERVATIONS`)

`app.agent4.identifiability.insufficient_observations_finding` compares the total number of
training data points (summed across every train-partition `Observation.values`) against the
number of fitted degrees of freedom (`len(targets)`). If there are fewer training points than
targets, the system is structurally underdetermined — Agent 4 refuses to even attempt
optimization (`CalibrationStatus.INSUFFICIENT_DATA`), since any resulting "estimate" would be an
arbitrary artifact of the optimizer's own starting point, never a meaningful fit.

**Establishes:** a hard, simple counting fact (points vs. free parameters).
**Does not establish:** sufficiency in the positive direction — having *more* points than
parameters is necessary, never sufficient, for a trustworthy fit (the other four checks above
exist precisely because point-counting alone cannot catch structural/practical
non-identifiability).

## How these combine into `CalibrationStatus`

Decided once, centrally, in `app.agent4.pipeline.run_agent4_pipeline` (mirroring
`app.agent3.report.determine_overall_status`'s own precedent of never letting two call sites
disagree about what "the run's status" means):

* Any `DEGENERATE_OBJECTIVE` finding → `UNIDENTIFIABLE`.
* Otherwise, any `WARNING`/`ERROR`-severity finding (a bound hit, low sensitivity, or high
  correlation) → `PARTIAL_SUCCESS`.
* Otherwise → `SUCCESS`.

A `SUCCESS` status means the optimizer converged, the objective improved, and none of Version
1's own modest checks found a reason for concern — it is not, and is never presented as, a
guarantee of correctness beyond what these five specific checks can see.
