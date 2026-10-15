"""Agent 4 version/policy markers.

These are architecture/contract-shape version markers, not claims of calibration maturity or
optimization correctness. Bump a constant only when the shape or policy it names actually
changes -- mirrors ``app.agent2.version``/``app.agent3.version``'s own identical convention on
the sibling repositories.

* ``AGENT4_CONTRACT_VERSION`` -- the version of Agent 4's own output contract
  (``app.agent4.types``: ``Agent4Report`` and the rest).
* ``AGENT4_HANDOFF_CONTRACT_VERSION`` -- the version of the Agent 2 model metadata embedded in
  this repository's own two-key handoff contract (``app.agent4.handoff``, ``docs/
  02_agent2_agent3_to_agent4_contract.md``). Five-Agent Workflow V1 Hardening increment: this
  is now ``"agent2-downstream-v1"``, the one canonical name Agent 2's own
  ``AGENT2_DOWNSTREAM_CONTRACT_VERSION`` emits and Agents 3, 4, and 5 all consume unchanged --
  replacing this repository's own previously-divergent ``"agent2-agent3-to-agent4-v1"`` name
  for what was already the identical real ``agent2_model`` artifact shape Agent 3 consumes. Not
  a guarantee that every upstream contract version is compatible -- version reconciliation is
  deferred, exactly as the Agent 1 -> Agent 2 and Agent 2 -> Agent 3 boundaries' own precedent.

  **Human-Readable Antimony Naming and Provenance Separation increment (Agent 2):** bumped
  ``"agent2-downstream-v1"`` -> ``"agent2-downstream-v2"``. Agent 2's generated Antimony
  symbols now prefer a curated biological name over a bare id, which this repository's own
  former ``app.agent4.antimony_ids`` module -- an independent re-implementation of Agent 2's
  *old* id-only naming rule -- could no longer resolve correctly. That module is removed
  entirely: Agent 2's own ``antimony_symbol_map`` field (new in v2) is the one source of truth
  for resolving a handoff entity id to its real, compiled Antimony identifier. See
  ``app.agent4.handoff.Agent2Model.antimony_symbol_map``.

  **Explicit Enzyme Concentration Representation increment (Agent 2):** bumped
  ``"agent2-downstream-v2"`` -> ``"agent2-downstream-v3"``, mirroring Agent 2's own real bump.
  Agent 2's handoff ``species[]`` gained ``source_protein_id`` (set only for an explicit
  catalytic-enzyme species) and ``kinetic_laws[]`` gained ``rate_law_scope``
  (``"EXPLICIT_ENZYME_CONCENTRATION"``/``"LUMPED_EFFECTIVE_RATE"``) -- both purely additive;
  this repository's own ``app.agent4.handoff`` parses every species/kinetic-law dict via
  ``.get(...)`` with no extra-key rejection, so the version bump alone (this repository does
  not yet read either new field) keeps parsing correct. Consuming them for Agent 4's own
  calibration-target eligibility is deferred to a future increment.
* ``ELIGIBILITY_POLICY_VERSION`` -- the version of the calibration-target eligibility policy
  (which provenance classes/initial conditions are calibratable by default, and how explicit
  authorization overrides that).
* ``OPTIMIZATION_POLICY_VERSION`` -- the version of the optimizer configuration policy (method,
  multi-start strategy, penalty for simulation failure, convergence tolerances).
* ``IDENTIFIABILITY_POLICY_VERSION`` -- the version of the practical-identifiability diagnostic
  rule set (bound-hit, weak-sensitivity, correlation, multi-start-flatness, insufficient-data
  thresholds).

**Calibration Evidence Integration into Weight-Space Exploration increment (cross-repo):**
``AGENT4_CONTRACT_VERSION`` bumped ``"0.1"`` -> ``"0.2"``. Purely additive: ``ParameterEstimate``
gained ``standard_error`` and ``CalibrationResult`` gained ``parameter_correlations`` (a new
``ParameterCorrelation`` type) -- both expose a local, approximate covariance estimate this
repository's own ``app.agent4.identifiability.correlation_findings`` already computed
internally (only to threshold-check its own ``HIGH_CORRELATION`` finding) but previously
discarded; no new calibration algorithm, no change to any existing finding/status/threshold
behavior. ``IDENTIFIABILITY_POLICY_VERSION`` is unchanged -- the rule set producing findings is
byte-identical; only new, additional fields were exposed alongside it. A downstream consumer
parsing via ``.get(...)`` (every sibling repository's own established convention) is
unaffected by either new field.
"""

from __future__ import annotations

AGENT4_CONTRACT_VERSION = "0.2"
AGENT4_HANDOFF_CONTRACT_VERSION = "agent2-downstream-v3"
ELIGIBILITY_POLICY_VERSION = "eligibility-v1"
OPTIMIZATION_POLICY_VERSION = "optimization-v1"
IDENTIFIABILITY_POLICY_VERSION = "identifiability-v1"

__all__ = [
    "AGENT4_CONTRACT_VERSION",
    "AGENT4_HANDOFF_CONTRACT_VERSION",
    "ELIGIBILITY_POLICY_VERSION",
    "IDENTIFIABILITY_POLICY_VERSION",
    "OPTIMIZATION_POLICY_VERSION",
]
