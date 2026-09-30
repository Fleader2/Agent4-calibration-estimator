"""Agent 4's own output contract: calibration request/result/report types.

Every type here is a plain, frozen, JSON-friendly dataclass -- consistent with every other
package in this project's own family. Numeric optimization/simulation data (objective values,
residuals, fitted parameter values) is kept as ``float``, not ``Decimal`` -- the same deliberate,
narrow divergence ``app.agent3.types`` documents: this data is produced by, and only ever
consumed by, real numerical ODE integration and nonlinear least-squares optimization (both IEEE
double precision throughout), so representing it as ``Decimal`` would imply a false exactness.
Agent 2's own contract values (parameter values, bounds) remain ``Decimal`` through
``app.agent4.handoff``; they are converted to ``float`` only at the point they are actually
handed to the simulation/optimization engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CalibrationTargetType(StrEnum):
    """What kind of model quantity one ``CalibrationTarget`` refers to."""

    PARAMETER = "PARAMETER"
    INITIAL_CONDITION = "INITIAL_CONDITION"


class ObservationKind(StrEnum):
    """Whether one ``Observation`` is a time-resolved series or a single steady-state value."""

    TIME_SERIES = "TIME_SERIES"
    STEADY_STATE = "STEADY_STATE"


class ObservationPartition(StrEnum):
    """Which half of the train/validation split one ``Observation`` belongs to.

    Assigned explicitly by the caller, or by ``app.agent4.observations.split_deterministically``
    -- never randomly and never silently; every ``Observation`` carries its own partition tag so
    a ``ResidualSummary`` can always be traced back to exactly which half of the data produced
    it.
    """

    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"


class CalibrationStatus(StrEnum):
    """The overall outcome of one Agent 4 calibration run.

    A small, closed, mutually-exclusive vocabulary, mirroring ``app.agent3.types
    .SimulationStatus``'s own precedent of never conflating qualitatively different situations:

    * ``INVALID_REQUEST`` -- the ``CalibrationRequest`` itself violates the eligibility policy
      (an unauthorized protected-provenance target, a ``fixed=True`` target, a target with no
      resolvable bounds) or references an id absent from the handoff. No simulation is ever
      attempted.
    * ``INSUFFICIENT_DATA`` -- the request is valid, but there are fewer usable training
      observations than fitted degrees of freedom; optimization is never attempted (a fit
      performed anyway would be meaningless, not merely uncertain).
    * ``SIMULATION_FAILED`` -- the baseline model (at its original, unperturbed parameter
      values) could not even be simulated once; optimization is never attempted.
    * ``OPTIMIZATION_FAILED`` -- every multi-start optimization attempt failed to converge.
    * ``UNIDENTIFIABLE`` -- the optimizer nominally converged, but the practical-identifiability
      diagnostics found the objective to be essentially flat across widely different parameter
      combinations (see ``app.agent4.identifiability``) -- the returned estimate is not
      trustworthy as a unique answer, even though a number was produced.
    * ``PARTIAL_SUCCESS`` -- optimization converged and materially improved the objective, but
      at least one identifiability finding (a bound hit, weak sensitivity, high correlation)
      warns the estimate should be treated with caution.
    * ``SUCCESS`` -- optimization converged, objective improved, and no identifiability concern
      was raised.
    """

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    OPTIMIZATION_FAILED = "OPTIMIZATION_FAILED"
    SIMULATION_FAILED = "SIMULATION_FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"
    INVALID_REQUEST = "INVALID_REQUEST"


class IdentifiabilitySeverity(StrEnum):
    """How seriously a ``IdentifiabilityFinding`` should be treated by a downstream consumer."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class IdentifiabilityCategory(StrEnum):
    """Which of Agent 4's own modest, practical identifiability checks a finding belongs to.
    Never a claim of formal structural identifiability -- see ``docs/
    05_identifiability_diagnostics.md`` for exactly what each category does and does not prove.
    """

    BOUND_HIT = "BOUND_HIT"
    LOW_SENSITIVITY = "LOW_SENSITIVITY"
    HIGH_CORRELATION = "HIGH_CORRELATION"
    DEGENERATE_OBJECTIVE = "DEGENERATE_OBJECTIVE"
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"


@dataclass(frozen=True, slots=True)
class CalibrationBounds:
    """Explicit lower/upper bounds for one calibration target.

    Never defaulted to an invented numeric range: if Agent 2's own declared
    ``lower_bound``/``upper_bound`` is available for a ``PARAMETER`` target and the request does
    not override it, that value is used; otherwise the request must supply this explicitly, or
    the target is rejected (``CalibrationStatus.INVALID_REQUEST``) -- see
    ``app.agent4.eligibility``.
    """

    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    """One quantity to calibrate: exactly one parameter or one species initial concentration.

    Listing a species as an ``INITIAL_CONDITION`` target **is itself** Version 1's own
    explicit-authorization mechanism for that species (see ``docs/
    03_calibration_target_eligibility_policy.md``) -- there is no separate "default-allowed"
    initial-condition class the way there is for heuristic/placeholder parameters.
    """

    target_kind: CalibrationTargetType
    target_id: str
    bounds: CalibrationBounds | None = None
    initial_guess: float | None = None
    prior_value: float | None = None
    prior_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class Observation:
    """One observed data series (time-resolved or steady-state) for exactly one species.

    ``sigma`` (per-point or scalar measurement uncertainty) drives the weighted-least-squares
    weight ``1/sigma**2``; omitted (``None``) means an unweighted (weight ``1.0``) contribution.
    Never fabricated by Agent 4 -- every value here is supplied by the caller.
    """

    observation_id: str
    target_species_id: str
    kind: ObservationKind
    partition: ObservationPartition
    time_points: tuple[float, ...]
    values: tuple[float, ...]
    sigma: tuple[float, ...] | float | None = None


@dataclass(frozen=True, slots=True)
class ObservationSet:
    """A named collection of ``Observation`` entries -- the complete, supplied experimental
    dataset one ``CalibrationRequest`` calibrates against. Never synthesized by Agent 4 itself
    outside of this repository's own clearly-labeled synthetic test fixtures."""

    observation_set_id: str
    observations: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Deterministic optimizer configuration.

    ``n_multistarts`` starting points are generated by a fixed, deterministic interpolation
    grid across each target's own bounds (see ``app.agent4.optimizer``) -- never a random draw,
    so no seed is needed for reproducibility; the same request always produces the same result.
    """

    method: str = "L-BFGS-B"
    n_multistarts: int = 3
    max_iterations: int = 200
    function_tolerance: float = 1e-10


@dataclass(frozen=True, slots=True)
class CalibrationRequest:
    """One complete, explicit request to calibrate a specific set of targets against a specific
    ``ObservationSet``.

    ``authorized_provenance_classes``/``authorized_parameter_ids`` are the **only** mechanisms
    that unlock a protected-provenance parameter (``LITERATURE_DERIVED``, ``CURATED``,
    ``AI_PREDICTED``, ``DERIVED_FROM_MACRO_KINETICS``, ``DERIVED_FROM_POOL_CONSERVATION``, or an
    already-``CALIBRATED`` one) as a target -- see ``app.agent4.eligibility``. A ``fixed=True``
    parameter can never be unlocked by any authorization; it is always rejected.
    """

    request_id: str
    targets: tuple[CalibrationTarget, ...]
    observations: ObservationSet
    optimizer: OptimizerConfig = OptimizerConfig()
    authorized_provenance_classes: tuple[str, ...] = ()
    authorized_parameter_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParameterEstimate:
    """The before/after record for one calibrated target.

    ``original_value``/``original_source`` are Agent 2's own already-declared value/provenance,
    carried through unchanged. ``new_source`` is always ``"CALIBRATED"`` -- the value Agent 2's
    own ``ParameterSource`` enum reserves specifically for Agent 4's own output (confirmed by
    direct inspection of ``agent2-antimony-builder``'s ``app/agent2/types.py`` this session).
    """

    target_kind: CalibrationTargetType
    target_id: str
    original_value: float | None
    original_source: str | None
    fitted_value: float
    lower_bound: float
    upper_bound: float
    at_lower_bound: bool
    at_upper_bound: bool
    new_source: str = "CALIBRATED"
    provenance_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResidualSummary:
    """Per-observation residual statistics for one partition (train or validation)."""

    observation_id: str
    partition: ObservationPartition
    residuals: tuple[float, ...]
    sum_squared_residual: float
    weighted_sum_squared_residual: float
    n_points: int


@dataclass(frozen=True, slots=True)
class IdentifiabilityFinding:
    """One structured, traceable observation about how trustworthy a fitted estimate is.

    Never a claim of formal/structural identifiability -- see ``docs/
    05_identifiability_diagnostics.md``.
    """

    finding_id: str
    category: IdentifiabilityCategory
    severity: IdentifiabilitySeverity
    summary: str
    explanation: str
    related_target_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """The full outcome of one optimization run: before/after objective, every parameter
    estimate, every residual summary, and every identifiability finding."""

    calibration_run_id: str
    status: CalibrationStatus
    objective_before: float | None
    objective_after: float | None
    parameter_estimates: tuple[ParameterEstimate, ...]
    residuals: tuple[ResidualSummary, ...]
    identifiability_findings: tuple[IdentifiabilityFinding, ...]
    optimizer_message: str | None
    n_multistarts: int
    best_start_index: int | None
    all_start_objectives: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Agent4Report:
    """The single, top-level structured output Agent 4 hands to Agent 5 (or a human reviewer)."""

    report_id: str
    agent4_contract_version: str
    agent2_model_id: str
    agent3_report_id: str
    request_id: str
    calibration: CalibrationResult
    fixed_parameters_verified_unchanged: bool
    fixed_parameter_ids_checked: tuple[str, ...]
    sufficient_for_agent5: bool
    sufficiency_note: str


__all__ = [
    "Agent4Report",
    "CalibrationBounds",
    "CalibrationRequest",
    "CalibrationResult",
    "CalibrationStatus",
    "CalibrationTarget",
    "CalibrationTargetType",
    "IdentifiabilityCategory",
    "IdentifiabilityFinding",
    "IdentifiabilitySeverity",
    "Observation",
    "ObservationKind",
    "ObservationPartition",
    "ObservationSet",
    "OptimizerConfig",
    "ParameterEstimate",
    "ResidualSummary",
]
