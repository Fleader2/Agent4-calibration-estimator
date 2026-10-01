"""The bounded real integration check against the current sce00061 Agent 2/Agent 3 artifacts.

Per the task's own explicit instruction: this proves Agent 4 can parse the real model and
Agent 3 report, and correctly identify calibration-eligible targets -- it never pretends to
calibrate biologically without suitable observation data, and never fabricates a yeast
observation merely to obtain a "successful" fit.
"""

from __future__ import annotations

from app.agent4.eligibility import (
    DEFAULT_CALIBRATABLE_PARAMETER_SOURCES,
    PROTECTED_PARAMETER_SOURCES,
)
from app.agent4.handoff import parse_agent4_handoff
from app.agent4.pipeline import run_agent4_pipeline
from app.agent4.types import (
    CalibrationBounds,
    CalibrationRequest,
    CalibrationStatus,
    CalibrationTarget,
    CalibrationTargetType,
    ObservationSet,
)
from app.agent4.version import AGENT4_HANDOFF_CONTRACT_VERSION
from tests.agent4.fixtures import load_real_sce00061_handoff


def test_real_handoff_declares_the_one_canonical_contract_version():
    """Five-Agent Workflow V1 Hardening compatibility check: the real sce00061 ``agent2_model``
    artifact, as produced by Agent 2's own canonical ``run_agent2_pipeline`` entrypoint, parses
    successfully and declares exactly the one canonical contract version this repository now
    expects -- with no re-stamping, no alias, and no per-consumer divergence from what Agent 3
    also consumes."""
    data = load_real_sce00061_handoff()
    assert (
        data["agent2_model"]["contract_version"]
        == AGENT4_HANDOFF_CONTRACT_VERSION
        == "agent2-downstream-v1"
    )
    parse_agent4_handoff(data)  # must not raise


def test_real_handoff_parses():
    handoff = parse_agent4_handoff(load_real_sce00061_handoff())
    # Counts reflect Agent 2's own canonical run_agent2_pipeline entrypoint (Five-Agent
    # Workflow V1 Hardening increment) -- richer than the pre-hardening partial-pipeline
    # artifact (109 parameters), since the canonical chain now also exercises reaction-context
    # resolution and enzyme-concentration/enzyme-state-dynamics resolution.
    assert len(handoff.agent2_model.parameters) == 115
    assert len(handoff.agent2_model.species) == 53
    assert handoff.agent3_diagnostics.overall_status == "STEADY_STATE_NOT_FOUND"


def test_real_heuristic_parameters_are_default_eligible_targets():
    handoff = parse_agent4_handoff(load_real_sce00061_handoff())
    heuristic_ids = {
        p.parameter_id
        for p in handoff.agent2_model.parameters
        if p.source in DEFAULT_CALIBRATABLE_PARAMETER_SOURCES
    }
    assert len(heuristic_ids) == 107
    assert heuristic_ids == set(handoff.agent3_diagnostics.heuristic_parameter_ids)

    protected_ids = {
        p.parameter_id
        for p in handoff.agent2_model.parameters
        if p.source in PROTECTED_PARAMETER_SOURCES
    }
    assert len(protected_ids) == 8  # 7 AI_PREDICTED + 1 LITERATURE_DERIVED
    assert len(heuristic_ids) + len(protected_ids) == 115


def test_real_missing_initial_conditions_are_all_53_species():
    handoff = parse_agent4_handoff(load_real_sce00061_handoff())
    missing_ic_species = [
        s.species_id
        for s in handoff.agent2_model.species
        if s.initial_amount is None and s.initial_concentration is None
    ]
    assert len(missing_ic_species) == 53

    missing_ic_finding = next(
        (
            f
            for f in handoff.agent3_diagnostics.diagnostics
            if f.finding_id == "missing-initial-conditions"
        ),
        None,
    )
    assert missing_ic_finding is not None
    assert set(missing_ic_finding.related_species_ids) == set(missing_ic_species)


def test_real_parameters_have_no_agent2_declared_bounds():
    """Confirms, with real data, that every one of the real parameters has
    lower_bound=upper_bound=None -- so calibrating any of them requires the caller to supply
    explicit bounds in the CalibrationRequest itself; Agent 4 never invents a numeric range."""
    handoff = parse_agent4_handoff(load_real_sce00061_handoff())
    with_bounds = [
        p
        for p in handoff.agent2_model.parameters
        if p.lower_bound is not None or p.upper_bound is not None
    ]
    assert with_bounds == []
    fixed_true = [p for p in handoff.agent2_model.parameters if p.fixed is True]
    assert fixed_true == []


def test_real_calibration_request_without_observations_is_honestly_blocked():
    """The central real-integration finding: Agent 4 can identify an eligible target and build
    a structurally valid request (explicit bounds supplied, since Agent 2 declares none), but
    with zero real observations supplied, calibration is correctly refused as
    INSUFFICIENT_DATA -- never a fabricated "successful" fit against invented yeast data."""
    handoff = parse_agent4_handoff(load_real_sce00061_handoff())
    heuristic_parameter = next(
        p
        for p in handoff.agent2_model.parameters
        if p.source in DEFAULT_CALIBRATABLE_PARAMETER_SOURCES
    )
    request = CalibrationRequest(
        request_id="real-sce00061-no-data-check",
        targets=(
            CalibrationTarget(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id=heuristic_parameter.parameter_id,
                bounds=CalibrationBounds(lower=1e-6, upper=1000.0),
            ),
        ),
        observations=ObservationSet(
            observation_set_id="no-real-observations-available", observations=()
        ),
    )
    report = run_agent4_pipeline(handoff, request)
    assert report.calibration.status is CalibrationStatus.INSUFFICIENT_DATA
    assert report.calibration.parameter_estimates == ()
    assert report.sufficient_for_agent5 is True
    assert "fewer training observations" in report.sufficiency_note
