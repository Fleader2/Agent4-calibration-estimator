"""Applying a candidate parameter/initial-condition vector and simulating it.

Every simulation here loads a **fresh** ``roadrunner.RoadRunner`` instance from the Agent 2
handoff's own unmodified ``antimony_text`` (via ``app.agent4.loader.load_antimony_model``) --
never reuses or resets a shared instance across optimizer evaluations. This satisfies the
task's own "every optimizer evaluation must use a fresh or safely reset simulation state"
requirement by always choosing "fresh": the small, deterministic models this Version 1 policy
targets compile in milliseconds, and a fresh load makes state leakage between evaluations
structurally impossible rather than merely avoided by discipline -- mirroring
``app.agent3.perturbation``'s own identical choice on the sibling repository.

A simulation/compilation failure for one candidate vector is **never** allowed to propagate as
an exception into the optimizer: ``evaluate_candidate`` catches every failure mode and reports
it as a ``CandidateSimulationOutcome`` with ``succeeded=False``, which
``app.agent4.objective`` turns into a large, fixed, finite penalty rather than crashing or
corrupting the optimizer's own internal state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.agent4.antimony_ids import candidate_parameter_antimony_id, candidate_species_antimony_id
from app.agent4.loader import load_antimony_model

#: Dense, fixed-resolution simulation grid used for every candidate evaluation -- trajectories
#: are linearly interpolated from this grid onto the exact observation time points, rather than
#: relying on any simulator-specific arbitrary-time-selection API. Deliberately generous (500
#: points) relative to the "modest" models this Version 1 policy targets; bump
#: ``OPTIMIZATION_POLICY_VERSION`` if this resolution ever changes.
_DENSE_GRID_POINTS = 500

#: Above this absolute magnitude, a finite simulated value is treated as divergent --
#: identical threshold and rationale to ``app.agent3.simulation.DIVERGENCE_MAGNITUDE_THRESHOLD``.
DIVERGENCE_MAGNITUDE_THRESHOLD = 1e12


@dataclass(frozen=True, slots=True)
class CandidateSimulationOutcome:
    """The outcome of simulating one candidate parameter/IC vector.

    ``species_trajectories``/``steady_state_values`` are keyed by the **original** Agent 2
    species id (never the sanitized Antimony identifier) -- resolved via
    ``app.agent4.antimony_ids`` before being returned, exactly as
    ``app.agent3.simulation``'s own ``id_map`` mechanism does.
    """

    succeeded: bool
    message: str | None
    time_grid: tuple[float, ...] | None = None
    species_trajectories: dict[str, tuple[float, ...]] | None = None
    steady_state_values: dict[str, float] | None = None


def _resolve_and_apply(rr, overrides: dict[str, float]) -> str | None:
    """Apply every ``{agent2_entity_id: value}`` override to the freshly-loaded model.

    Tries the parameter-id sanitization first, then the species-id sanitization, for each
    override key -- an override key is always either a parameter id or a species id, and
    Agent 4 never guesses which without checking against the model's own real identifiers.
    Returns ``None`` on success, or a human-readable message naming the first unresolved key.
    """
    real_parameter_ids = set(rr.model.getGlobalParameterIds())
    real_species_ids = set(rr.model.getFloatingSpeciesIds())
    for entity_id, value in overrides.items():
        param_candidate = candidate_parameter_antimony_id(entity_id)
        species_candidate = candidate_species_antimony_id(entity_id)
        if param_candidate in real_parameter_ids:
            rr[param_candidate] = value
        elif species_candidate in real_species_ids:
            rr[species_candidate] = value
        else:
            return (
                f"Could not resolve override target {entity_id!r} against the loaded model "
                f"(tried parameter id {param_candidate!r} and species id "
                f"{species_candidate!r})."
            )
    return None


def evaluate_candidate(
    antimony_text: str,
    overrides: dict[str, float],
    time_points_needed: tuple[float, ...],
    *,
    species_id_map: dict[str, str] | None = None,
    steady_state_needed: bool = False,
) -> CandidateSimulationOutcome:
    """Load a fresh model, apply ``overrides``, and simulate it.

    ``time_points_needed`` is the sorted, de-duplicated union of every observation time point
    this candidate must be compared against; simulation runs on a dense, fixed-resolution grid
    from ``0`` to ``max(time_points_needed)`` and every species trajectory is linearly
    interpolated onto exactly those requested points before being returned.

    ``species_id_map`` (``{antimony_species_id: original_agent2_species_id}``, typically
    ``app.agent4.antimony_ids.build_species_antimony_id_map``'s own output) relabels every
    trajectory/steady-state key back to the original Agent 2 species id -- without it, keys are
    RoadRunner's own real Antimony identifiers verbatim, which would break provenance tracing
    for a real Agent 2 handoff (see ``app.agent3.simulation``'s own identical ``id_map``
    mechanism and rationale on the sibling repository).
    """
    load_result = load_antimony_model(antimony_text)
    if not load_result.succeeded or load_result.roadrunner_instance is None:
        return CandidateSimulationOutcome(succeeded=False, message=load_result.message)

    rr = load_result.roadrunner_instance
    species_ids = list(rr.model.getFloatingSpeciesIds())
    id_map = species_id_map or {}
    resolution_error = _resolve_and_apply(rr, overrides)
    if resolution_error is not None:
        return CandidateSimulationOutcome(succeeded=False, message=resolution_error)

    end_time = max(time_points_needed) if time_points_needed else 1.0
    if end_time <= 0:
        end_time = 1.0

    try:
        raw = rr.simulate(0.0, end_time, _DENSE_GRID_POINTS)
    except Exception as exc:  # pragma: no cover - roadrunner raises plain Exception/RuntimeError
        return CandidateSimulationOutcome(
            succeeded=False, message=f"Integration failed for this candidate: {exc}"
        )

    array = np.asarray(raw)
    colnames = list(raw.colnames)
    grid_times = array[:, 0].astype(float)

    flat_values = array[:, 1:].astype(float).ravel()
    if np.isnan(flat_values).any() or np.isinf(flat_values).any():
        return CandidateSimulationOutcome(
            succeeded=False, message="Candidate simulation produced a NaN or Inf value."
        )
    finite_values = flat_values[np.isfinite(flat_values)]
    if finite_values.size and np.abs(finite_values).max() > DIVERGENCE_MAGNITUDE_THRESHOLD:
        return CandidateSimulationOutcome(
            succeeded=False,
            message=(
                f"Candidate simulation exceeded the divergence threshold "
                f"({DIVERGENCE_MAGNITUDE_THRESHOLD:g})."
            ),
        )

    trajectories: dict[str, tuple[float, ...]] = {}
    requested = np.asarray(time_points_needed, dtype=float) if time_points_needed else grid_times
    for col_index, col_name in enumerate(colnames[1:], start=1):
        raw_id = col_name.strip("[]")
        species_id = id_map.get(raw_id, raw_id)
        series = array[:, col_index].astype(float)
        interpolated = np.interp(requested, grid_times, series)
        trajectories[species_id] = tuple(float(v) for v in interpolated)

    steady_state_values: dict[str, float] | None = None
    if steady_state_needed:
        try:
            residual = rr.steadyState()
        except Exception as exc:
            return CandidateSimulationOutcome(
                succeeded=False, message=f"Steady-state solver raised for this candidate: {exc}"
            )
        if not math.isfinite(residual):
            return CandidateSimulationOutcome(
                succeeded=False,
                message=f"Steady-state solver returned a non-finite residual: {residual!r}.",
            )
        steady_state_values = {
            id_map.get(sid, sid): float(value)
            for sid, value in zip(
                species_ids, rr.model.getFloatingSpeciesConcentrations(), strict=True
            )
        }

    return CandidateSimulationOutcome(
        succeeded=True,
        message=None,
        time_grid=tuple(float(v) for v in requested),
        species_trajectories=trajectories,
        steady_state_values=steady_state_values,
    )


__all__ = [
    "DIVERGENCE_MAGNITUDE_THRESHOLD",
    "CandidateSimulationOutcome",
    "evaluate_candidate",
]
