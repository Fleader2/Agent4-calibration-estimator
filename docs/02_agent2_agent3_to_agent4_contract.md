# Agent 2/Agent 3 → Agent 4 Contract (`agent2-agent3-to-agent4-v1`)

This is the smallest stable input shape Agent 4 Version 1 requires, expressed as a plain,
JSON-serializable dict with two top-level keys: `agent2_model` and `agent3_diagnostics`.
`app.agent4.handoff.parse_agent4_handoff` parses and validates it into the frozen `Agent4Handoff`
dataclass family in `app.agent4.handoff`. Agent 4 never imports Agent 2's or Agent 3's Python
package — this mirrors the established decoupling pattern at every other boundary in this
project.

## Why two dicts, not one

`agent2_model` and `agent3_diagnostics` are two distinct upstream artifacts with two distinct
purposes:

* **`agent2_model`** — the actual, compilable `antimony_text`, plus every structural/provenance
  fact needed to select and bound a calibration target.
* **`agent3_diagnostics`** — a lightweight mirror of the parts of Agent 3's own `Agent3Report`
  relevant to choosing and interpreting calibration targets (e.g. which species Agent 3's own
  `missing-initial-conditions` finding flagged).

## A genuinely new contract, not a modification of Agent 2 or Agent 3

**Important fact, confirmed by direct code inspection this session:** Agent 2's own
`ParameterSpecification` type (`agent2-antimony-builder`, `app/agent2/types.py`) already carries
`lower_bound`, `upper_bound`, and `fixed` fields, and `SpeciesSpecification` already carries an
`initialization_source` field — but the sibling Agent 3 repository's own Version 1 handoff
contract (`app.agent3.handoff.HandoffParameter`/`HandoffSpecies`) does **not** carry any of these
four fields, because Agent 3's own job (diagnostics) never needed them.

`agent2_model` in this contract is therefore a deliberate **superset** of Agent 3's own
`Agent2Handoff` shape: every field Agent 3's contract has, plus these four additional fields.
This is a new, Agent-4-specific contract — it does not require, and was not achieved by, any
change to Agent 2's or Agent 3's existing code or output. No function in `agent2-antimony-
builder` currently emits this exact shape automatically (there is no `translate_to_agent4`); a
caller (or this repository's own real-integration fixture-building script) assembles it from
Agent 2's already-existing `ParameterSpecification`/`SpeciesSpecification` fields plus Agent 3's
already-existing `Agent3Report` fields.

## `agent2_model`

Identical field-for-field to Agent 3's own `agent2_model` shape (see `agent3-simulation-
diagnostics/docs/02_agent2_to_agent3_contract.md` for the full compartments/species/reactions/
kinetic_laws/model_assumptions table), **plus**:

| Field (on `parameters[]`) | Type | Notes |
|---|---|---|
| `lower_bound` | decimal-like or null | Agent 2's own declared lower bound, if any. |
| `upper_bound` | decimal-like or null | Agent 2's own declared upper bound, if any. |
| `fixed` | bool or null | `true` is an absolute veto on calibration — see `docs/03_calibration_target_eligibility_policy.md`. |

| Field (on `species[]`) | Type | Notes |
|---|---|---|
| `initialization_source` | str or null | A `ParameterSource` value (e.g. `"CURATED"`), or null when Agent 2 never set one. |

## `agent3_diagnostics`

| Field | Type | Notes |
|---|---|---|
| `report_id` | str | Agent 3's own report id, carried through onto `Agent4Report.agent3_report_id`. |
| `agent3_contract_version` | str | Agent 3's own contract version marker. |
| `agent2_model_id` | str | Cross-check against `agent2_model.model_id`. |
| `overall_status` | str | Agent 3's own `SimulationStatus` value (e.g. `"STEADY_STATE_NOT_FOUND"`). |
| `diagnostics[]` | list of finding views | See below. |
| `heuristic_parameter_ids` | list[str] | Agent 3's own classification — carried through, never re-derived. |
| `placeholder_parameter_ids` | list[str] | Agent 3's own classification — carried through, never re-derived. |
| `agent2_unresolved_reaction_ids` | list[str] | Carried through from Agent 3's own report. |
| `agent2_unresolved_kinetic_law_ids` | list[str] | Carried through from Agent 3's own report. |
| `sufficient_for_agent4` | bool | Agent 3's own verdict on whether its report was sufficient for Agent 4 — informational only. |

Each `diagnostics[]` entry is a lightweight mirror of one Agent 3 `DiagnosticFinding`:
`finding_id`, `category`, `severity`, `summary`, `related_species_ids`, `related_reaction_ids`,
`related_parameter_ids`. Agent 4 reads these (e.g. to cross-check which species Agent 3's own
`missing-initial-conditions` finding flagged) but never recomputes Agent 3's own diagnostic
logic itself.

## How the real `sce00061` fixture for this repository was built

`tests/fixtures/sce00061_agent4_handoff.json` was assembled, in this session, by combining:

1. The sibling Agent 3 repository's own real `tests/fixtures/sce00061_agent2_handoff.json`
   (already-validated real Agent 2 output for the current `sce00061` model: 53 species, 38
   reactions, 109 parameters).
2. A real extract of Agent 2's own `lower_bound`/`upper_bound`/`fixed`/`initialization_source`
   fields for that same model, taken from an already-saved pilot artifact
   (`agent1-biochemical-curator/artifacts/pilots/.../25_parameter_declaration.json` and
   `45_full_reactions_and_species.json`) — **confirmed, with real data, that all 109 parameters
   have `lower_bound=upper_bound=fixed=None` and all 53 species have
   `initialization_source=None`** for the current model: no increment in the pipeline populates
   these fields yet. This is an honest, disclosed characteristic of the current model, not a
   fixture-building error.
3. A fresh, real `Agent3Report` produced by actually running Agent 3's own committed pipeline
   (`run_agent3_pipeline`) against artifact (1) — never fabricated or guessed.

No new Agent 1/Agent 2/Agent 3 pipeline run was performed to build this fixture beyond the one
Agent 3 pipeline execution in step 3; no yeast-specific biological data was added anywhere in
this process.
