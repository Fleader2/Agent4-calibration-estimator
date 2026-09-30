# Agent 4 — Responsibility Boundary

Agent 4 is the **Calibration and Parameter Estimation Agent** in the five-agent biochemical-
modeling pipeline (Agent 1 curation → Agent 2 model assembly → Agent 3 simulation/diagnostics →
**Agent 4 calibration** → Agent 5, not yet built).

## Central rule

> Agent 4 may estimate uncertain parameters and initial conditions from supplied observation
> data, but must never overwrite curated/evidence-backed values or invent observations.

Every design decision in this repository traces back to this one sentence.

## What Agent 4 does

1. **Consumes** a plain, JSON-serializable handoff dict combining Agent 2's own model/parameter
   metadata and a lightweight mirror of Agent 3's diagnostic report — see `docs/
   02_agent2_agent3_to_agent4_contract.md`.
2. **Validates** an explicit `CalibrationRequest` against the calibration-target eligibility
   policy (`docs/03_calibration_target_eligibility_policy.md`) before ever attempting a
   simulation — an unauthorized or malformed request is refused (`INVALID_REQUEST`), never
   silently narrowed or partially honored.
3. **Simulates** the unmodified Agent 2 Antimony model (via the same antimony/libRoadRunner
   stack Agent 3 already validated) at candidate parameter/initial-condition vectors, always
   from a **fresh** model load — never a shared, mutated instance.
4. **Optimizes** a weighted-least-squares objective via deterministic, bounded, multi-start
   `scipy.optimize.minimize`, never introducing randomness that would make two runs of the same
   request disagree.
5. **Reports** objective value and every parameter estimate before and after fitting, residuals
   for both train and validation partitions, and a small set of practical-identifiability
   findings — see `docs/05_identifiability_diagnostics.md`.
6. **Preserves provenance**: every fitted value's original value and original Agent 2 source are
   carried through unchanged alongside the new, `CALIBRATED`-tagged value — see `docs/
   04_provenance_semantics.md`.
7. **Verifies** that no parameter Agent 2 declared `fixed=True` was ever touched, as a defensive,
   literal invariant check (not merely "we never call code that would touch it").

## What Agent 4 explicitly does NOT do

- **Does not overwrite an evidence-backed parameter's value** unless the request explicitly
  authorizes that specific protected provenance class or parameter id — and even then, the
  *original* value/source is retained alongside the fitted one, never replaced in place.
- **Does not invent observations.** Every `Observation` in an `ObservationSet` is supplied by the
  caller; Agent 4 never synthesizes a data point to help a fit along. The only "synthetic" data
  anywhere in this repository is in its own clearly-labeled test fixtures, built from known-
  ground-truth toy models, never presented as real biological data.
- **Does not invent a numeric bound.** A target's bounds come from the request itself, or from
  Agent 2's own already-declared `lower_bound`/`upper_bound` (when both are set) — if neither is
  available, the target is rejected, not defaulted to some arbitrary range.
- **Does not calibrate a `fixed=True` parameter**, ever, regardless of any authorization — this
  is an absolute veto with no override anywhere in the request contract.
- **Does not perform formal structural identifiability analysis.** Every identifiability check
  in Version 1 is a local, practical, approximate heuristic — see `docs/
  05_identifiability_diagnostics.md` for exactly what is and is not established.
- **Does not use Bayesian/MCMC methods** in Version 1 — a deterministic, bounded,
  multi-start gradient optimizer (`scipy.optimize.minimize`) is sufficient for this stage's own
  modest scope, and avoids an unnecessarily heavy dependency.
- **Does not modify Agent 1, Agent 2, or Agent 3 code or output.** Agent 4 never imports any
  sibling package's Python code — it consumes only the plain-dict contract documented in `docs/
  02_agent2_agent3_to_agent4_contract.md`, mirroring the established decoupling pattern used at
  every other boundary in this project.
- **Does not decide whether a fit is scientifically good.** `CalibrationStatus` and
  `sufficient_for_agent5` are completeness/traceability judgments ("did the optimizer converge,"
  "is this report structurally useful to Agent 5"), never a claim that the resulting biology is
  correct.
- **Does not pretend to calibrate yeast biology without real data.** The real `sce00061`
  integration in this repository deliberately stops at "target identified, bounds resolvable,
  request well-formed" and reports `INSUFFICIENT_DATA` rather than fabricating an observation to
  produce a "successful" fit.

## Relationship to neighboring agents

| Agent | Relationship to Agent 4 |
|---|---|
| Agent 1 (curation) | Three hops upstream; Agent 4 never reads Agent 1 output directly. |
| Agent 2 (model assembly) | Indirectly upstream via the handoff's `agent2_model` section — Agent 4 never imports Agent 2's Python package. |
| Agent 3 (simulation/diagnostics) | Indirectly upstream via the handoff's `agent3_diagnostics` section — Agent 4 never imports Agent 3's Python package, and never re-derives a diagnostic Agent 3 already computed; it only reads Agent 3's own already-declared findings. |
| Agent 4 (this repository) | Calibrates only authorized targets against supplied observations; never edits the model structure. |
| Agent 5 (not yet built) | Directly downstream. Consumes `Agent4Report` to decide what further action (e.g. model revision, additional experiments) is warranted — Agent 4 hands it a traceable calibration outcome, not a final verdict on the model's quality. |
