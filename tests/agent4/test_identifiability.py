"""Tests for ``app.agent4.identifiability``: practical, local, modest identifiability checks."""

from __future__ import annotations

import numpy as np

from app.agent4.antimony_ids import build_species_antimony_id_map
from app.agent4.identifiability import (
    bound_hit_findings,
    correlation_findings,
    degenerate_objective_findings,
    insufficient_observations_finding,
    is_at_bound,
    low_sensitivity_findings,
)
from app.agent4.optimizer import StartResult
from app.agent4.types import (
    CalibrationTarget,
    CalibrationTargetType,
    IdentifiabilityCategory,
    Observation,
    ObservationKind,
    ObservationPartition,
)
from tests.agent4.fixtures import (
    TWO_PARAMETER_ANTIMONY,
    UNIDENTIFIABLE_ANTIMONY,
    UNIDENTIFIABLE_TRUE_PRODUCT,
    minimal_agent2_model_dict,
    parsed_handoff,
    simulate_truth,
)


def test_is_at_bound_detects_lower_and_upper():
    assert is_at_bound(0.001, 0.0, 10.0) == (True, False)
    assert is_at_bound(9.999, 0.0, 10.0) == (False, True)
    assert is_at_bound(5.0, 0.0, 10.0) == (False, False)


def test_bound_hit_findings_flags_only_hit_targets():
    findings = bound_hit_findings(["a", "b"], [True, False], [False, False])
    assert len(findings) == 1
    assert findings[0].related_target_ids == ("a",)


def test_insufficient_observations_finding_triggers_below_threshold():
    finding = insufficient_observations_finding(n_train_points=1, n_free_parameters=2)
    assert finding is not None
    assert finding.category.name == "INSUFFICIENT_OBSERVATIONS"


def test_insufficient_observations_finding_none_when_enough_data():
    assert insufficient_observations_finding(n_train_points=10, n_free_parameters=2) is None


def test_degenerate_objective_findings_flags_widely_different_equally_good_fits():
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    results = [
        StartResult(
            start_vector=(0.1, 0.1),
            final_vector=(0.1, 0.9),
            objective=0.001,
            converged=True,
            message="ok",
        ),
        StartResult(
            start_vector=(0.9, 0.9),
            final_vector=(0.9, 0.1),
            objective=0.001,
            converged=True,
            message="ok",
        ),
    ]
    findings = degenerate_objective_findings(results, bounds)
    assert len(findings) == 1
    assert findings[0].category is IdentifiabilityCategory.DEGENERATE_OBJECTIVE


def test_degenerate_objective_findings_empty_when_fits_agree():
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    results = [
        StartResult(
            start_vector=(0.1, 0.1),
            final_vector=(0.5, 0.5),
            objective=0.001,
            converged=True,
            message="ok",
        ),
        StartResult(
            start_vector=(0.9, 0.9),
            final_vector=(0.501, 0.499),
            objective=0.0011,
            converged=True,
            message="ok",
        ),
    ]
    assert degenerate_objective_findings(results, bounds) == ()


def test_low_sensitivity_findings_flags_unused_parameter():
    model_dict = minimal_agent2_model_dict(
        TWO_PARAMETER_ANTIMONY,
        species=[
            {
                "species_id": "S1",
                "name": "S1",
                "compartment_id": "C",
                "initial_concentration": "10",
            },
            {"species_id": "S2", "name": "S2", "compartment_id": "C", "initial_concentration": "0"},
            {"species_id": "S3", "name": "S3", "compartment_id": "C", "initial_concentration": "0"},
        ],
        parameters=[
            {
                "parameter_id": "k1",
                "name": "k1",
                "source": "HEURISTIC_INITIALIZATION",
                "value": "0.5",
            },
            {
                "parameter_id": "k2",
                "name": "k2",
                "source": "HEURISTIC_INITIALIZATION",
                "value": "0.2",
            },
        ],
    )
    handoff = parsed_handoff(model_dict)
    time_points, values = simulate_truth(TWO_PARAMETER_ANTIMONY, "s_S1", end_time=2, num_points=5)
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
    targets = (
        CalibrationTarget(target_kind=CalibrationTargetType.PARAMETER, target_id="k1"),
        CalibrationTarget(target_kind=CalibrationTargetType.PARAMETER, target_id="k2"),
    )
    id_map = build_species_antimony_id_map(["S1", "S2", "S3"])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    best_vector = np.array([0.5, 0.2])
    from app.agent4.objective import evaluate_vector

    baseline = evaluate_vector(handoff.agent2_model, targets, best_vector, obs, id_map)
    findings = low_sensitivity_findings(
        handoff.agent2_model, targets, best_vector, bounds, obs, id_map, baseline.objective
    )
    related = {t for f in findings for t in f.related_target_ids}
    assert "k2" in related


def test_correlation_findings_flags_unidentifiable_pair():
    model_dict = minimal_agent2_model_dict(
        UNIDENTIFIABLE_ANTIMONY,
        species=[
            {
                "species_id": "S1",
                "name": "S1",
                "compartment_id": "C",
                "initial_concentration": "10",
            },
            {"species_id": "S2", "name": "S2", "compartment_id": "C", "initial_concentration": "0"},
        ],
        parameters=[
            {
                "parameter_id": "k1",
                "name": "k1",
                "source": "HEURISTIC_INITIALIZATION",
                "value": str(UNIDENTIFIABLE_TRUE_PRODUCT),
            },
            {
                "parameter_id": "k2",
                "name": "k2",
                "source": "HEURISTIC_INITIALIZATION",
                "value": "1.0",
            },
        ],
    )
    handoff = parsed_handoff(model_dict)
    time_points, values = simulate_truth(UNIDENTIFIABLE_ANTIMONY, "s_S1", end_time=5, num_points=6)
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
    targets = (
        CalibrationTarget(target_kind=CalibrationTargetType.PARAMETER, target_id="k1"),
        CalibrationTarget(target_kind=CalibrationTargetType.PARAMETER, target_id="k2"),
    )
    id_map = build_species_antimony_id_map(["S1", "S2"])
    bounds = [(0.01, 2.0), (0.01, 2.0)]
    best_vector = np.array([UNIDENTIFIABLE_TRUE_PRODUCT, 1.0])
    findings = correlation_findings(handoff.agent2_model, targets, best_vector, bounds, obs, id_map)
    assert len(findings) == 1
    assert findings[0].category is IdentifiabilityCategory.HIGH_CORRELATION
