# Provenance Semantics

How Agent 4 preserves, and never destroys, the provenance trail Agent 1 → Agent 2 → Agent 3
already built.

## Before/after, never in-place replacement

Every `ParameterEstimate` (`app.agent4.types.ParameterEstimate`) carries **both** the original
value/source and the fitted value/new-source side by side:

| Field | Meaning |
|---|---|
| `original_value` | Agent 2's own already-declared value (parameter `.value`, or species `.initial_concentration`/`.initial_amount`) — `None` if Agent 2 never declared one (e.g. a missing initial condition). |
| `original_source` | Agent 2's own already-declared `ParameterSource` string (or species `initialization_source`) — `None` if Agent 2 never declared one. |
| `fitted_value` | The value Agent 4's own optimizer converged to. |
| `new_source` | Always the literal string `"CALIBRATED"` — the exact value Agent 2's own `ParameterSource` enum reserves specifically for Agent 4's output (confirmed by direct inspection of `agent2-antimony-builder`'s `app/agent2/types.py` this session: `ParameterSpecification` raises if Agent 2 code itself ever tries to declare a parameter with this source). |
| `provenance_refs` | `(observation_set_id, calibration_run_id)` — traces the fitted value back to exactly which supplied dataset and which optimization run produced it. |

Nothing is ever overwritten in place: a downstream consumer (Agent 5, or a human reviewer) sees
the complete history of a value in one record, never just its final number.

## Why `new_source` is always `"CALIBRATED"`, never conditional

A calibrated value is always tagged `CALIBRATED`, regardless of what its original source was
(even if it was itself already `CALIBRATED` from a prior run, and the request explicitly
authorized recalibrating it). This is deliberate: the new value's provenance is "this specific
Agent 4 run produced it," full stop — the fact that its *predecessor* was, say,
`HEURISTIC_INITIALIZATION` is preserved separately in `original_source`, never merged into or
confused with the new value's own provenance tag.

## Fixed-parameter protection is verified, not merely assumed

`Agent4Report.fixed_parameters_verified_unchanged` and `.fixed_parameter_ids_checked` exist so a
downstream consumer does not have to trust "the eligibility policy should have prevented this" —
`app.agent4.pipeline` computes the full list of `fixed=True` parameter ids from the handoff
itself and raises `Agent4Error` (a hard failure, not a silently-dropped finding) if any of them
is ever found among the produced estimates. In every current test and the real integration run,
`fixed_parameters_verified_unchanged` is `True` and the check list matches Agent 2's own declared
`fixed=True` parameters exactly (currently empty for the real `sce00061` model, since no
parameter there is marked fixed).

## Immutability of original Agent 2/Agent 3 artifacts

* `Agent4Handoff` is parsed once, from a plain dict, into frozen (`@dataclass(frozen=True)`)
  dataclasses — nothing in `app.agent4` ever mutates a parsed handoff object.
* Every simulation (`app.agent4.simulation.evaluate_candidate`) loads a **fresh**
  `roadrunner.RoadRunner` instance from the handoff's own unmodified `antimony_text` string —
  the string itself is only ever read, never written, and no shared, mutated instance is reused
  across optimizer evaluations, multi-starts, or targets.
* No file on disk that represents Agent 2's or Agent 3's own output is ever written to by this
  repository's runtime code (only this repository's own test-fixture-building scripts, run once
  during development, touched those files — and only to *read* them, writing a brand-new file
  under Agent 4's own `tests/fixtures/` directory).

## What "provenance-preserving" does NOT mean

It does not mean Agent 4 refuses to change a value — calibration's entire purpose is to change
values. It means every change is fully attributable: what the value was, why it was eligible to
change, what produced the new value, and exactly which observations and optimization run are
responsible — never a silent overwrite with no trace of what came before or why.
