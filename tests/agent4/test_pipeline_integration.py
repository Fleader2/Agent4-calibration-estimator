"""End-to-end pipeline tests against the synthetic calibration fixtures the task requires:

1. a simple known one-parameter model where the true value is recovered
2. a two-parameter model with identifiable parameters
3. a deliberately unidentifiable model
4. a case with a missing initial condition fitted successfully
5. a case proving curated/fixed parameters remain unchanged
6. a simulation-failure-during-optimization case handled safely
7. deterministic repeatability
8. train/validation split behavior
9. provenance traceability
"""

from __future__ import annotations

import pytest

from app.agent4.observations import split_deterministically
from app.agent4.pipeline import run_agent4_pipeline
from app.agent4.types import (
    CalibrationBounds,
    CalibrationRequest,
    CalibrationStatus,
    CalibrationTarget,
    CalibrationTargetType,
    Observation,
    ObservationKind,
    ObservationPartition,
    ObservationSet,
    OptimizerConfig,
)
from tests.agent4.fixtures import (
    MISSING_IC_ANTIMONY,
    MISSING_IC_KNOWN_K,
    MISSING_IC_TRUE_VALUE,
    ONE_PARAMETER_ANTIMONY,
    ONE_PARAMETER_TRUE_K,
    TWO_PARAMETER_ANTIMONY,
    TWO_PARAMETER_TRUE_K1,
    TWO_PARAMETER_TRUE_K2,
    UNIDENTIFIABLE_ANTIMONY,
    minimal_agent2_model_dict,
    parsed_handoff,
    simulate_truth,
)


def _time_series_obs(
    observation_id,
    species_id,
    time_points,
    values,
    partition=ObservationPartition.TRAIN,
    sigma=None,
):
    return Observation(
        observation_id=observation_id,
        target_species_id=species_id,
        kind=ObservationKind.TIME_SERIES,
        partition=partition,
        time_points=time_points,
        values=values,
        sigma=sigma,
    )


# --- 1. one-parameter recovery -------------------------------------------------------------


def _one_parameter_report():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    time_points, values = simulate_truth(ONE_PARAMETER_ANTIMONY, "s_S1", end_time=10, num_points=11)
    obs_set = ObservationSet(
        observation_set_id="synthetic-one-parameter-decay",
        observations=(_time_series_obs("o1", "S1", time_points, values),),
    )
    request = CalibrationRequest(
        request_id="req-1",
        targets=(
            CalibrationTarget(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id="k1",
                bounds=CalibrationBounds(lower=0.01, upper=5.0),
            ),
        ),
        observations=obs_set,
        optimizer=OptimizerConfig(n_multistarts=3),
    )
    return run_agent4_pipeline(handoff, request), handoff, request


def test_one_parameter_recovery():
    report, _, _ = _one_parameter_report()
    assert report.calibration.status in (
        CalibrationStatus.SUCCESS,
        CalibrationStatus.PARTIAL_SUCCESS,
    )
    estimate = report.calibration.parameter_estimates[0]
    assert estimate.fitted_value == pytest.approx(ONE_PARAMETER_TRUE_K, rel=1e-2)
    assert report.calibration.objective_after < report.calibration.objective_before


# --- 2. two-parameter identifiable recovery ------------------------------------------------


def test_two_parameter_identifiable_recovery():
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
                "value": "0.1",
            },
            {
                "parameter_id": "k2",
                "name": "k2",
                "source": "HEURISTIC_INITIALIZATION",
                "value": "0.1",
            },
        ],
    )
    handoff = parsed_handoff(model_dict)
    time_points, s1 = simulate_truth(TWO_PARAMETER_ANTIMONY, "s_S1", end_time=15, num_points=16)
    _, s2 = simulate_truth(TWO_PARAMETER_ANTIMONY, "s_S2", end_time=15, num_points=16)
    _, s3 = simulate_truth(TWO_PARAMETER_ANTIMONY, "s_S3", end_time=15, num_points=16)

    obs_set = ObservationSet(
        observation_set_id="synthetic-two-parameter-chain",
        observations=(
            _time_series_obs("o-s1", "S1", time_points, s1),
            _time_series_obs("o-s2", "S2", time_points, s2),
            _time_series_obs("o-s3", "S3", time_points, s3),
        ),
    )
    request = CalibrationRequest(
        request_id="req-2",
        targets=(
            CalibrationTarget(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id="k1",
                bounds=CalibrationBounds(lower=0.01, upper=2.0),
            ),
            CalibrationTarget(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id="k2",
                bounds=CalibrationBounds(lower=0.01, upper=2.0),
            ),
        ),
        observations=obs_set,
        optimizer=OptimizerConfig(n_multistarts=4),
    )
    report = run_agent4_pipeline(handoff, request)
    assert report.calibration.status in (
        CalibrationStatus.SUCCESS,
        CalibrationStatus.PARTIAL_SUCCESS,
    )
    estimates = {e.target_id: e.fitted_value for e in report.calibration.parameter_estimates}
    assert estimates["k1"] == pytest.approx(TWO_PARAMETER_TRUE_K1, rel=5e-2)
    assert estimates["k2"] == pytest.approx(TWO_PARAMETER_TRUE_K2, rel=5e-2)


# --- 3. deliberately unidentifiable model ---------------------------------------------------


def test_deliberately_unidentifiable_model_flagged():
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
                "value": "0.1",
            },
            {
                "parameter_id": "k2",
                "name": "k2",
                "source": "HEURISTIC_INITIALIZATION",
                "value": "0.1",
            },
        ],
    )
    handoff = parsed_handoff(model_dict)
    time_points, values = simulate_truth(UNIDENTIFIABLE_ANTIMONY, "s_S1", end_time=5, num_points=6)
    obs_set = ObservationSet(
        observation_set_id="synthetic-unidentifiable",
        observations=(_time_series_obs("o1", "S1", time_points, values),),
    )
    request = CalibrationRequest(
        request_id="req-3",
        targets=(
            CalibrationTarget(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id="k1",
                bounds=CalibrationBounds(lower=0.01, upper=2.0),
            ),
            CalibrationTarget(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id="k2",
                bounds=CalibrationBounds(lower=0.01, upper=2.0),
            ),
        ),
        observations=obs_set,
        optimizer=OptimizerConfig(n_multistarts=4),
    )
    report = run_agent4_pipeline(handoff, request)
    # Either explicitly UNIDENTIFIABLE, or the correlation finding directly names the pair --
    # either way the practical-identifiability machinery must surface *something* here.
    categories = {f.category.value for f in report.calibration.identifiability_findings}
    assert (
        report.calibration.status is CalibrationStatus.UNIDENTIFIABLE
        or "HIGH_CORRELATION" in categories
    )


# --- 4. missing initial condition fitted successfully ---------------------------------------


def test_missing_initial_condition_fitted_successfully():
    model_dict = minimal_agent2_model_dict(
        MISSING_IC_ANTIMONY,
        species=[
            {
                "species_id": "S1",
                "name": "S1",
                "compartment_id": "C",
                "initial_amount": None,
                "initial_concentration": None,
            }
        ],
        parameters=[
            {
                "parameter_id": "k1",
                "name": "k1",
                "source": "CURATED",
                "value": str(MISSING_IC_KNOWN_K),
            }
        ],
    )
    handoff = parsed_handoff(model_dict)
    assert handoff.agent2_model.species[0].initial_amount is None
    assert handoff.agent2_model.species[0].initial_concentration is None

    true_antimony = MISSING_IC_ANTIMONY.replace("s_S1 = 0", f"s_S1 = {MISSING_IC_TRUE_VALUE}")
    time_points, values = simulate_truth(true_antimony, "s_S1", end_time=8, num_points=9)
    obs_set = ObservationSet(
        observation_set_id="synthetic-missing-ic",
        observations=(_time_series_obs("o1", "S1", time_points, values),),
    )
    request = CalibrationRequest(
        request_id="req-4",
        targets=(
            CalibrationTarget(
                target_kind=CalibrationTargetType.INITIAL_CONDITION,
                target_id="S1",
                bounds=CalibrationBounds(lower=0.0, upper=20.0),
            ),
        ),
        observations=obs_set,
        optimizer=OptimizerConfig(n_multistarts=3),
    )
    report = run_agent4_pipeline(handoff, request)
    assert report.calibration.status in (
        CalibrationStatus.SUCCESS,
        CalibrationStatus.PARTIAL_SUCCESS,
    )
    estimate = report.calibration.parameter_estimates[0]
    assert estimate.target_kind is CalibrationTargetType.INITIAL_CONDITION
    assert estimate.original_value is None
    assert estimate.fitted_value == pytest.approx(MISSING_IC_TRUE_VALUE, rel=5e-2)


# --- 5. curated/fixed parameters remain unchanged --------------------------------------------


def test_curated_parameter_omitted_from_targets_is_never_touched():
    report, _, _ = _one_parameter_report()
    # k1 (HEURISTIC) was the only target; a hypothetical second, CURATED parameter never
    # appearing in targets at all is never estimated.
    estimated_ids = {e.target_id for e in report.calibration.parameter_estimates}
    assert estimated_ids == {"k1"}


def test_fixed_parameter_request_rejected_even_with_authorization():
    model_dict = minimal_agent2_model_dict(
        ONE_PARAMETER_ANTIMONY,
        parameters=[
            {
                "parameter_id": "k1",
                "name": "k1",
                "source": "CURATED",
                "value": "0.7",
                "fixed": True,
            }
        ],
    )
    handoff = parsed_handoff(model_dict)
    obs_set = ObservationSet(
        observation_set_id="synthetic-fixed-param",
        observations=(_time_series_obs("o1", "S1", (0.0, 1.0), (10.0, 5.0)),),
    )
    request = CalibrationRequest(
        request_id="req-5",
        targets=(
            CalibrationTarget(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id="k1",
                bounds=CalibrationBounds(lower=0.0, upper=1.0),
            ),
        ),
        observations=obs_set,
        authorized_provenance_classes=("CURATED",),
        authorized_parameter_ids=("k1",),
    )
    report = run_agent4_pipeline(handoff, request)
    assert report.calibration.status is CalibrationStatus.INVALID_REQUEST
    assert report.calibration.parameter_estimates == ()
    assert report.fixed_parameter_ids_checked == ("k1",)


def test_protected_parameter_without_authorization_rejected():
    model_dict = minimal_agent2_model_dict(
        ONE_PARAMETER_ANTIMONY,
        parameters=[
            {"parameter_id": "k1", "name": "k1", "source": "LITERATURE_DERIVED", "value": "0.7"}
        ],
    )
    handoff = parsed_handoff(model_dict)
    obs_set = ObservationSet(
        observation_set_id="synthetic-protected-param",
        observations=(_time_series_obs("o1", "S1", (0.0, 1.0), (10.0, 5.0)),),
    )
    request = CalibrationRequest(
        request_id="req-5b",
        targets=(
            CalibrationTarget(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id="k1",
                bounds=CalibrationBounds(lower=0.0, upper=1.0),
            ),
        ),
        observations=obs_set,
    )
    report = run_agent4_pipeline(handoff, request)
    assert report.calibration.status is CalibrationStatus.INVALID_REQUEST
    assert report.calibration.parameter_estimates == ()


# --- 6. simulation-failure-during-optimization handled safely -------------------------------


def test_simulation_failure_during_optimization_handled_safely():
    antimony_text = """
model overflow_risk
  compartment C = 1
  species s_S1 in C
  s_S1 = 1
  p_k1 = 2.0
  J1: -> s_S1;
  J1 = p_k1 * p_k1
end
"""
    model_dict = minimal_agent2_model_dict(
        antimony_text,
        parameters=[
            {
                "parameter_id": "k1",
                "name": "k1",
                "source": "HEURISTIC_INITIALIZATION",
                "value": "2.0",
            }
        ],
    )
    handoff = parsed_handoff(model_dict)
    obs_set = ObservationSet(
        observation_set_id="synthetic-sim-failure",
        observations=(_time_series_obs("o1", "S1", (0.0, 1.0, 2.0), (1.0, 5.0, 9.0)),),
    )
    request = CalibrationRequest(
        request_id="req-6",
        targets=(
            CalibrationTarget(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id="k1",
                bounds=CalibrationBounds(lower=0.01, upper=1e160),
                initial_guess=2.0,
            ),
        ),
        observations=obs_set,
        optimizer=OptimizerConfig(n_multistarts=3),
    )
    # Must not raise -- a candidate deep in [0.01, 1e160] will overflow p_k1*p_k1 to inf.
    report = run_agent4_pipeline(handoff, request)
    assert isinstance(report.calibration.status, CalibrationStatus)
    if report.calibration.objective_after is not None:
        import math

        assert math.isfinite(report.calibration.objective_after)


# --- 7. deterministic repeatability ----------------------------------------------------------


def test_deterministic_repeatability():
    report1, _, _ = _one_parameter_report()
    report2, _, _ = _one_parameter_report()
    assert report1.calibration.status == report2.calibration.status
    assert report1.calibration.objective_before == report2.calibration.objective_before
    assert report1.calibration.objective_after == report2.calibration.objective_after
    est1 = report1.calibration.parameter_estimates[0].fitted_value
    est2 = report2.calibration.parameter_estimates[0].fitted_value
    assert est1 == est2


# --- 8. train/validation split behavior -------------------------------------------------------


def test_train_validation_split_behavior():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    time_points, values = simulate_truth(ONE_PARAMETER_ANTIMONY, "s_S1", end_time=10, num_points=11)
    observations = tuple(
        _time_series_obs(f"o{i}", "S1", (t,), (v,))
        for i, (t, v) in enumerate(zip(time_points, values, strict=True))
    )
    split_observations = split_deterministically(observations, validation_fraction=0.3)
    n_validation = sum(
        1 for o in split_observations if o.partition is ObservationPartition.VALIDATION
    )
    n_train = sum(1 for o in split_observations if o.partition is ObservationPartition.TRAIN)
    assert n_validation == round(len(observations) * 0.3)
    assert n_train + n_validation == len(observations)

    obs_set = ObservationSet(
        observation_set_id="synthetic-train-val-split", observations=split_observations
    )
    request = CalibrationRequest(
        request_id="req-8",
        targets=(
            CalibrationTarget(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id="k1",
                bounds=CalibrationBounds(lower=0.01, upper=5.0),
            ),
        ),
        observations=obs_set,
        optimizer=OptimizerConfig(n_multistarts=3),
    )
    report = run_agent4_pipeline(handoff, request)
    partitions_in_residuals = {r.partition for r in report.calibration.residuals}
    assert ObservationPartition.TRAIN in partitions_in_residuals
    assert ObservationPartition.VALIDATION in partitions_in_residuals
    n_residual_train = sum(
        1 for r in report.calibration.residuals if r.partition is ObservationPartition.TRAIN
    )
    n_residual_validation = sum(
        1 for r in report.calibration.residuals if r.partition is ObservationPartition.VALIDATION
    )
    assert n_residual_train == n_train
    assert n_residual_validation == n_validation


# --- 9. provenance traceability ----------------------------------------------------------------


def test_provenance_traceability():
    report, handoff, request = _one_parameter_report()
    estimate = report.calibration.parameter_estimates[0]
    assert estimate.original_value == pytest.approx(0.1)
    assert estimate.original_source == "HEURISTIC_INITIALIZATION"
    assert estimate.new_source == "CALIBRATED"
    assert request.observations.observation_set_id in estimate.provenance_refs
    assert report.calibration.calibration_run_id in estimate.provenance_refs
    assert report.agent2_model_id == handoff.agent2_model.model_id
    assert report.agent3_report_id == handoff.agent3_diagnostics.report_id
