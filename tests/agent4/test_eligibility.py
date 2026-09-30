"""Tests for ``app.agent4.eligibility``: the calibration-target eligibility policy."""

from __future__ import annotations

from app.agent4.eligibility import (
    check_initial_condition_eligible,
    check_parameter_eligible,
    resolve_bounds,
    validate_request,
)
from app.agent4.types import (
    CalibrationBounds,
    CalibrationRequest,
    CalibrationTarget,
    CalibrationTargetType,
    Observation,
    ObservationKind,
    ObservationPartition,
    ObservationSet,
)
from tests.agent4.fixtures import ONE_PARAMETER_ANTIMONY, minimal_agent2_model_dict, parsed_handoff


def _request(targets, observations=(), **overrides):
    obs_set = ObservationSet(observation_set_id="obs-set-1", observations=observations)
    return CalibrationRequest(
        request_id="req-1", targets=targets, observations=obs_set, **overrides
    )


def test_heuristic_parameter_eligible_by_default():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    param = handoff.agent2_model.parameters[0]
    assert param.source == "HEURISTIC_INITIALIZATION"
    request = _request(())
    assert check_parameter_eligible(param, request) is None


def test_placeholder_and_default_sources_eligible_by_default():
    handoff = parsed_handoff(
        minimal_agent2_model_dict(
            ONE_PARAMETER_ANTIMONY,
            parameters=[
                {
                    "parameter_id": "k1",
                    "name": "k1",
                    "source": "PLACEHOLDER",
                    "value": "0.1",
                }
            ],
        )
    )
    request = _request(())
    assert check_parameter_eligible(handoff.agent2_model.parameters[0], request) is None


def test_curated_parameter_ineligible_without_authorization():
    handoff = parsed_handoff(
        minimal_agent2_model_dict(
            ONE_PARAMETER_ANTIMONY,
            parameters=[{"parameter_id": "k1", "name": "k1", "source": "CURATED", "value": "0.1"}],
        )
    )
    request = _request(())
    reason = check_parameter_eligible(handoff.agent2_model.parameters[0], request)
    assert reason is not None
    assert "protected source" in reason


def test_curated_parameter_eligible_with_class_authorization():
    handoff = parsed_handoff(
        minimal_agent2_model_dict(
            ONE_PARAMETER_ANTIMONY,
            parameters=[{"parameter_id": "k1", "name": "k1", "source": "CURATED", "value": "0.1"}],
        )
    )
    request = _request((), authorized_provenance_classes=("CURATED",))
    assert check_parameter_eligible(handoff.agent2_model.parameters[0], request) is None


def test_curated_parameter_eligible_with_explicit_id_authorization():
    handoff = parsed_handoff(
        minimal_agent2_model_dict(
            ONE_PARAMETER_ANTIMONY,
            parameters=[{"parameter_id": "k1", "name": "k1", "source": "CURATED", "value": "0.1"}],
        )
    )
    request = _request((), authorized_parameter_ids=("k1",))
    assert check_parameter_eligible(handoff.agent2_model.parameters[0], request) is None


def test_fixed_parameter_ineligible_even_with_authorization():
    handoff = parsed_handoff(
        minimal_agent2_model_dict(
            ONE_PARAMETER_ANTIMONY,
            parameters=[
                {
                    "parameter_id": "k1",
                    "name": "k1",
                    "source": "HEURISTIC_INITIALIZATION",
                    "value": "0.1",
                    "fixed": True,
                }
            ],
        )
    )
    request = _request(
        (),
        authorized_parameter_ids=("k1",),
        authorized_provenance_classes=("HEURISTIC_INITIALIZATION",),
    )
    reason = check_parameter_eligible(handoff.agent2_model.parameters[0], request)
    assert reason is not None
    assert "fixed=True" in reason


def test_initial_condition_eligible_by_being_listed_as_a_target():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    species = handoff.agent2_model.species[0]
    assert species.initialization_source is None
    request = _request(())
    assert check_initial_condition_eligible(species, request) is None


def test_initial_condition_with_protected_initialization_source_requires_authorization():
    handoff = parsed_handoff(
        minimal_agent2_model_dict(
            ONE_PARAMETER_ANTIMONY,
            species=[
                {
                    "species_id": "S1",
                    "name": "S1",
                    "compartment_id": "C",
                    "initial_concentration": "10",
                    "initialization_source": "CURATED",
                }
            ],
        )
    )
    species = handoff.agent2_model.species[0]
    request = _request(())
    assert check_initial_condition_eligible(species, request) is not None
    authorized_request = _request((), authorized_provenance_classes=("CURATED",))
    assert check_initial_condition_eligible(species, authorized_request) is None


def test_resolve_bounds_uses_request_bounds_first():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    target = CalibrationTarget(
        target_kind=CalibrationTargetType.PARAMETER,
        target_id="k1",
        bounds=CalibrationBounds(lower=0.0, upper=1.0),
    )
    bounds = resolve_bounds(target, handoff.agent2_model)
    assert bounds == CalibrationBounds(lower=0.0, upper=1.0)


def test_resolve_bounds_falls_back_to_agent2_declared_bounds():
    handoff = parsed_handoff(
        minimal_agent2_model_dict(
            ONE_PARAMETER_ANTIMONY,
            parameters=[
                {
                    "parameter_id": "k1",
                    "name": "k1",
                    "source": "HEURISTIC_INITIALIZATION",
                    "value": "0.1",
                    "lower_bound": "0.01",
                    "upper_bound": "2.0",
                }
            ],
        )
    )
    target = CalibrationTarget(target_kind=CalibrationTargetType.PARAMETER, target_id="k1")
    bounds = resolve_bounds(target, handoff.agent2_model)
    assert bounds == CalibrationBounds(lower=0.01, upper=2.0)


def test_resolve_bounds_none_when_unresolvable():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    target = CalibrationTarget(target_kind=CalibrationTargetType.PARAMETER, target_id="k1")
    assert resolve_bounds(target, handoff.agent2_model) is None


def test_validate_request_rejects_missing_bounds():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    target = CalibrationTarget(target_kind=CalibrationTargetType.PARAMETER, target_id="k1")
    obs = Observation(
        observation_id="o1",
        target_species_id="S1",
        kind=ObservationKind.TIME_SERIES,
        partition=ObservationPartition.TRAIN,
        time_points=(0.0, 1.0),
        values=(10.0, 5.0),
    )
    request = _request((target,), (obs,))
    violations = validate_request(handoff.agent2_model, request)
    assert len(violations) == 1
    assert "bound" in violations[0].reason.lower()


def test_validate_request_accepts_well_formed_request():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    target = CalibrationTarget(
        target_kind=CalibrationTargetType.PARAMETER,
        target_id="k1",
        bounds=CalibrationBounds(lower=0.01, upper=5.0),
    )
    obs = Observation(
        observation_id="o1",
        target_species_id="S1",
        kind=ObservationKind.TIME_SERIES,
        partition=ObservationPartition.TRAIN,
        time_points=(0.0, 1.0),
        values=(10.0, 5.0),
    )
    request = _request((target,), (obs,))
    assert validate_request(handoff.agent2_model, request) == ()


def test_validate_request_rejects_empty_targets():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    request = _request(())
    violations = validate_request(handoff.agent2_model, request)
    assert len(violations) == 1


def test_validate_request_rejects_unknown_parameter_id():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    target = CalibrationTarget(
        target_kind=CalibrationTargetType.PARAMETER,
        target_id="does-not-exist",
        bounds=CalibrationBounds(lower=0.0, upper=1.0),
    )
    request = _request((target,))
    violations = validate_request(handoff.agent2_model, request)
    assert any("exists" in v.reason for v in violations)


def test_validate_request_rejects_duplicate_targets():
    handoff = parsed_handoff(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    target = CalibrationTarget(
        target_kind=CalibrationTargetType.PARAMETER,
        target_id="k1",
        bounds=CalibrationBounds(lower=0.0, upper=1.0),
    )
    request = _request((target, target))
    violations = validate_request(handoff.agent2_model, request)
    assert any("Duplicate" in v.reason for v in violations)
