"""Tests for ``app.agent4.optimizer``: deterministic multi-start optimization."""

from __future__ import annotations

import numpy as np
import pytest

from app.agent4.antimony_ids import build_species_antimony_id_map
from app.agent4.optimizer import generate_start_vectors, run_multistart_optimization
from app.agent4.types import (
    CalibrationTarget,
    CalibrationTargetType,
    Observation,
    ObservationKind,
    ObservationPartition,
    OptimizerConfig,
)
from tests.agent4.fixtures import (
    ONE_PARAMETER_ANTIMONY,
    ONE_PARAMETER_TRUE_K,
    minimal_agent2_model_dict,
    parsed_handoff,
    simulate_truth,
)


def test_generate_start_vectors_is_deterministic():
    baseline = np.array([0.5, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    first = generate_start_vectors(baseline, bounds, 4)
    second = generate_start_vectors(baseline, bounds, 4)
    assert [tuple(v) for v in first] == [tuple(v) for v in second]


def test_generate_start_vectors_respects_bounds():
    baseline = np.array([50.0])
    bounds = [(0.0, 1.0)]
    starts = generate_start_vectors(baseline, bounds, 5)
    for start in starts:
        assert 0.0 <= start[0] <= 1.0


def test_generate_start_vectors_first_start_is_clipped_baseline():
    baseline = np.array([50.0])
    bounds = [(0.0, 1.0)]
    starts = generate_start_vectors(baseline, bounds, 3)
    assert starts[0][0] == 1.0


def test_multistart_recovers_true_parameter():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    time_points, values = simulate_truth(ONE_PARAMETER_ANTIMONY, "s_S1", end_time=5, num_points=11)
    obs = (
        Observation(
            observation_id="o1",
            target_species_id="S1",
            kind=ObservationKind.TIME_SERIES,
            partition=ObservationPartition.TRAIN,
            time_points=time_points,
            values=values,
        ),
    )
    targets = (CalibrationTarget(target_kind=CalibrationTargetType.PARAMETER, target_id="k1"),)
    id_map = build_species_antimony_id_map(["S1"])
    results = run_multistart_optimization(
        handoff.agent2_model,
        targets,
        [(0.01, 5.0)],
        np.array([0.1]),
        obs,
        id_map,
        OptimizerConfig(n_multistarts=3),
    )
    best = min((r for r in results if r.converged), key=lambda r: r.objective)
    assert best.final_vector[0] == pytest.approx(ONE_PARAMETER_TRUE_K, rel=1e-2)
