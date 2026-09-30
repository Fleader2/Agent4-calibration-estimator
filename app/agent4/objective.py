"""The Version 1 weighted-least-squares calibration objective.

``J(theta) = sum_over_observations sum_over_points w * (y_sim(t; theta) - y_obs(t))**2
             + sum_over_targets prior_weight * (theta - prior_value)**2``

A candidate vector that fails to simulate (compilation failure, solver exception, NaN/Inf,
divergence) never raises into the optimizer and never silently returns ``0``/``nan`` -- it
returns ``PENALTY_OBJECTIVE_VALUE`` (a large, fixed, finite constant), so a gradient-based
optimizer always sees a well-defined, if very bad, objective value and is pushed away from that
region rather than crashing or being corrupted by a non-finite value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.agent4.handoff import Agent2Model
from app.agent4.simulation import evaluate_candidate
from app.agent4.types import CalibrationTarget, Observation, ObservationKind

#: A candidate that fails to simulate is penalized with this objective value -- large enough to
#: never be mistaken for a legitimate fit (orders of magnitude above any realistic weighted SSE
#: for the "modest" models this Version 1 policy targets), but finite, so gradient-based
#: optimizers never receive ``inf``/``nan``. Bump ``OPTIMIZATION_POLICY_VERSION`` if this value
#: ever changes.
PENALTY_OBJECTIVE_VALUE = 1e12


@dataclass(frozen=True, slots=True)
class ObservationResidual:
    """Per-point residuals for one ``Observation`` at one candidate vector."""

    observation_id: str
    partition: str
    residuals: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ObjectiveEvaluation:
    """The full outcome of evaluating one candidate vector against one set of observations."""

    objective: float
    succeeded: bool
    message: str | None
    residuals: tuple[ObservationResidual, ...]


def weights_for_observation(observation: Observation) -> np.ndarray:
    n = len(observation.values)
    if observation.sigma is None:
        return np.ones(n)
    if isinstance(observation.sigma, (int, float)):
        return np.full(n, 1.0 / (float(observation.sigma) ** 2))
    sigma_arr = np.asarray(observation.sigma, dtype=float)
    return 1.0 / (sigma_arr**2)


def _union_time_points(observations: tuple[Observation, ...]) -> tuple[float, ...]:
    points: set[float] = set()
    for obs in observations:
        if obs.kind is ObservationKind.TIME_SERIES:
            points.update(obs.time_points)
    return tuple(sorted(points))


def evaluate_vector(
    agent2_model: Agent2Model,
    targets: tuple[CalibrationTarget, ...],
    vector: np.ndarray,
    observations: tuple[Observation, ...],
    species_id_map: dict[str, str],
) -> ObjectiveEvaluation:
    """Evaluate one candidate ``vector`` (in the same order as ``targets``) against
    ``observations``, returning the weighted-least-squares objective plus every per-observation
    residual. Never raises for a simulation failure -- see this module's own docstring."""
    overrides = {
        target.target_id: float(value) for target, value in zip(targets, vector, strict=True)
    }
    needs_steady_state = any(obs.kind is ObservationKind.STEADY_STATE for obs in observations)
    union_times = _union_time_points(observations)

    outcome = evaluate_candidate(
        agent2_model.antimony_text,
        overrides,
        union_times,
        species_id_map=species_id_map,
        steady_state_needed=needs_steady_state,
    )
    if not outcome.succeeded:
        return ObjectiveEvaluation(
            objective=PENALTY_OBJECTIVE_VALUE,
            succeeded=False,
            message=outcome.message,
            residuals=(),
        )

    time_index = {t: i for i, t in enumerate(union_times)}
    total = 0.0
    residual_records: list[ObservationResidual] = []

    for obs in observations:
        weights = weights_for_observation(obs)
        if obs.kind is ObservationKind.STEADY_STATE:
            simulated = outcome.steady_state_values or {}
            sim_value = simulated.get(obs.target_species_id)
            if sim_value is None:
                return ObjectiveEvaluation(
                    objective=PENALTY_OBJECTIVE_VALUE,
                    succeeded=False,
                    message=(
                        f"Species {obs.target_species_id!r} not present in steady-state result."
                    ),
                    residuals=(),
                )
            residuals = np.array([sim_value - obs.values[0]])
        else:
            trajectory = (outcome.species_trajectories or {}).get(obs.target_species_id)
            if trajectory is None:
                return ObjectiveEvaluation(
                    objective=PENALTY_OBJECTIVE_VALUE,
                    succeeded=False,
                    message=(
                        f"Species {obs.target_species_id!r} not present in simulated trajectory."
                    ),
                    residuals=(),
                )
            indices = [time_index[t] for t in obs.time_points]
            simulated_values = np.array([trajectory[i] for i in indices])
            residuals = simulated_values - np.asarray(obs.values, dtype=float)

        total += float(np.sum(weights * residuals**2))
        residual_records.append(
            ObservationResidual(
                observation_id=obs.observation_id,
                partition=obs.partition.value,
                residuals=tuple(float(r) for r in residuals),
            )
        )

    for target, value in zip(targets, vector, strict=True):
        if target.prior_value is not None and target.prior_weight > 0:
            total += target.prior_weight * (float(value) - target.prior_value) ** 2

    if not np.isfinite(total):
        return ObjectiveEvaluation(
            objective=PENALTY_OBJECTIVE_VALUE,
            succeeded=False,
            message="Objective evaluated to a non-finite value.",
            residuals=(),
        )

    return ObjectiveEvaluation(
        objective=total, succeeded=True, message=None, residuals=tuple(residual_records)
    )


__all__ = [
    "PENALTY_OBJECTIVE_VALUE",
    "ObjectiveEvaluation",
    "ObservationResidual",
    "evaluate_vector",
    "weights_for_observation",
]
