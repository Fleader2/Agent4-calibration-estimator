"""Agent 4 version/policy markers.

These are architecture/contract-shape version markers, not claims of calibration maturity or
optimization correctness. Bump a constant only when the shape or policy it names actually
changes -- mirrors ``app.agent2.version``/``app.agent3.version``'s own identical convention on
the sibling repositories.

* ``AGENT4_CONTRACT_VERSION`` -- the version of Agent 4's own output contract
  (``app.agent4.types``: ``Agent4Report`` and the rest).
* ``AGENT4_HANDOFF_CONTRACT_VERSION`` -- the version of the Agent 2/Agent 3 -> Agent 4 handoff
  contract this repository was built against (``app.agent4.handoff``, ``docs/
  02_agent2_agent3_to_agent4_contract.md``). Not a guarantee that every upstream contract
  version is compatible -- version reconciliation is deferred, exactly as the Agent 1 -> Agent 2
  and Agent 2 -> Agent 3 boundaries' own precedent.
* ``ELIGIBILITY_POLICY_VERSION`` -- the version of the calibration-target eligibility policy
  (which provenance classes/initial conditions are calibratable by default, and how explicit
  authorization overrides that).
* ``OPTIMIZATION_POLICY_VERSION`` -- the version of the optimizer configuration policy (method,
  multi-start strategy, penalty for simulation failure, convergence tolerances).
* ``IDENTIFIABILITY_POLICY_VERSION`` -- the version of the practical-identifiability diagnostic
  rule set (bound-hit, weak-sensitivity, correlation, multi-start-flatness, insufficient-data
  thresholds).
"""

from __future__ import annotations

AGENT4_CONTRACT_VERSION = "0.1"
AGENT4_HANDOFF_CONTRACT_VERSION = "agent2-agent3-to-agent4-v1"
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
