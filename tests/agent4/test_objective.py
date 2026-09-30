"""Tests for ``app.agent4.objective``: the weighted-least-squares objective."""

from __future__ import annotations

import numpy as np
import pytest

from app.agent4.antimony_ids import build_species_antimony_id_map
from app.agent4.objective import PENALTY_OBJECTIVE_VALUE, evaluate_vector
from app.agent4.types import (
    CalibrationTarget,
    CalibrationTargetType,
    Observation,
    ObservationKind,
    ObservationPartition,
)
from tests.agent4.fixtures import (
    ONE_PARAMETER_ANTIMONY,
    ONE_PARAMETER_TRUE_K,
    minimal_agent2_model_dict,
    parsed_handoff,
    simulate_truth,
)


def _target(bounds=None):
    return (CalibrationTarget(target_kind=CalibrationTargetType.PARAMETER, target_id="k1"),)


def test_exact_match_gives_zero_objective():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    time_points, values = simulate_truth(ONE_PARAMETER_ANTIMONY, "s_S1", end_time=5, num_points=6)
    obs = Observation(
        observation_id="o1",
        target_species_id="S1",
        kind=ObservationKind.TIME_SERIES,
        partition=ObservationPartition.TRAIN,
        time_points=time_points,
        values=values,
    )
    id_map = build_species_antimony_id_map(["S1"])
    result = evaluate_vector(
        handoff.agent2_model, _target(), np.array([ONE_PARAMETER_TRUE_K]), (obs,), id_map
    )
    assert result.succeeded
    assert result.objective < 1e-6


def test_wrong_value_gives_positive_objective():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    time_points, values = simulate_truth(ONE_PARAMETER_ANTIMONY, "s_S1", end_time=5, num_points=6)
    obs = Observation(
        observation_id="o1",
        target_species_id="S1",
        kind=ObservationKind.TIME_SERIES,
        partition=ObservationPartition.TRAIN,
        time_points=time_points,
        values=values,
    )
    id_map = build_species_antimony_id_map(["S1"])
    result = evaluate_vector(handoff.agent2_model, _target(), np.array([0.1]), (obs,), id_map)
    assert result.succeeded
    assert result.objective > 1.0


def test_sigma_weighting_scales_objective():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    time_points, values = simulate_truth(ONE_PARAMETER_ANTIMONY, "s_S1", end_time=5, num_points=6)
    id_map = build_species_antimony_id_map(["S1"])

    unweighted = Observation(
        observation_id="o1",
        target_species_id="S1",
        kind=ObservationKind.TIME_SERIES,
        partition=ObservationPartition.TRAIN,
        time_points=time_points,
        values=values,
        sigma=None,
    )
    weighted = Observation(
        observation_id="o1",
        target_species_id="S1",
        kind=ObservationKind.TIME_SERIES,
        partition=ObservationPartition.TRAIN,
        time_points=time_points,
        values=values,
        sigma=2.0,
    )
    vector = np.array([0.1])
    unweighted_result = evaluate_vector(
        handoff.agent2_model, _target(), vector, (unweighted,), id_map
    )
    weighted_result = evaluate_vector(handoff.agent2_model, _target(), vector, (weighted,), id_map)
    assert weighted_result.objective == pytest.approx(unweighted_result.objective / 4.0)


def test_invalid_antimony_returns_penalty_not_exception():
    handoff = parsed_handoff(minimal_agent2_model_dict("this is not $$$ valid antimony"))
    obs = Observation(
        observation_id="o1",
        target_species_id="S1",
        kind=ObservationKind.TIME_SERIES,
        partition=ObservationPartition.TRAIN,
        time_points=(0.0, 1.0),
        values=(10.0, 5.0),
    )
    id_map = build_species_antimony_id_map(["S1"])
    result = evaluate_vector(handoff.agent2_model, _target(), np.array([0.1]), (obs,), id_map)
    assert not result.succeeded
    assert result.objective == PENALTY_OBJECTIVE_VALUE


def test_steady_state_observation_objective():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    obs = Observation(
        observation_id="o1",
        target_species_id="S1",
        kind=ObservationKind.STEADY_STATE,
        partition=ObservationPartition.TRAIN,
        time_points=(),
        values=(0.0,),
    )
    id_map = build_species_antimony_id_map(["S1"])
    result = evaluate_vector(
        handoff.agent2_model, _target(), np.array([ONE_PARAMETER_TRUE_K]), (obs,), id_map
    )
    assert result.succeeded
    assert result.objective < 1e-6


def test_prior_penalty_added_when_configured():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    time_points, values = simulate_truth(ONE_PARAMETER_ANTIMONY, "s_S1", end_time=5, num_points=6)
    obs = Observation(
        observation_id="o1",
        target_species_id="S1",
        kind=ObservationKind.TIME_SERIES,
        partition=ObservationPartition.TRAIN,
        time_points=time_points,
        values=values,
    )
    id_map = build_species_antimony_id_map(["S1"])
    target_with_prior = (
        CalibrationTarget(
            target_kind=CalibrationTargetType.PARAMETER,
            target_id="k1",
            prior_value=0.0,
            prior_weight=1.0,
        ),
    )
    no_prior_result = evaluate_vector(
        handoff.agent2_model, _target(), np.array([ONE_PARAMETER_TRUE_K]), (obs,), id_map
    )
    prior_result = evaluate_vector(
        handoff.agent2_model, target_with_prior, np.array([ONE_PARAMETER_TRUE_K]), (obs,), id_map
    )
    assert prior_result.objective > no_prior_result.objective
    assert prior_result.objective == pytest.approx(
        no_prior_result.objective + ONE_PARAMETER_TRUE_K**2
    )
