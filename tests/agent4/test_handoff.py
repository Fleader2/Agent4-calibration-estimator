"""Contract tests for ``app.agent4.handoff``."""

from __future__ import annotations

import pytest

from app.agent4.errors import HandoffContractError
from app.agent4.handoff import parse_agent4_handoff
from tests.agent4.fixtures import (
    ONE_PARAMETER_ANTIMONY,
    build_handoff_dict,
    minimal_agent2_model_dict,
)


def test_valid_handoff_parses():
    handoff = parse_agent4_handoff(
        build_handoff_dict(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    )
    assert handoff.agent2_model.model_id == "test-model-1"
    assert len(handoff.agent2_model.parameters) == 1
    assert handoff.agent3_diagnostics.report_id == "agent3-report-test"


def test_missing_required_field_raises():
    data = build_handoff_dict(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    del data["agent2_model"]["antimony_text"]
    with pytest.raises(HandoffContractError):
        parse_agent4_handoff(data)


def test_not_a_dict_raises():
    with pytest.raises(HandoffContractError):
        parse_agent4_handoff("not a dict")


def test_missing_agent2_model_key_raises():
    data = {"agent3_diagnostics": {}}
    with pytest.raises(HandoffContractError):
        parse_agent4_handoff(data)


def test_missing_agent3_diagnostics_key_raises():
    data = {"agent2_model": minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY)}
    with pytest.raises(HandoffContractError):
        parse_agent4_handoff(data)


def test_parameter_bounds_and_fixed_round_trip():
    model = minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY)
    model["parameters"][0]["lower_bound"] = "0.01"
    model["parameters"][0]["upper_bound"] = "5.0"
    model["parameters"][0]["fixed"] = True
    handoff = parse_agent4_handoff(build_handoff_dict(model))
    param = handoff.agent2_model.parameters[0]
    assert float(param.lower_bound) == pytest.approx(0.01)
    assert float(param.upper_bound) == pytest.approx(5.0)
    assert param.fixed is True


def test_species_initialization_source_round_trip():
    model = minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY)
    model["species"][0]["initialization_source"] = "CURATED"
    handoff = parse_agent4_handoff(build_handoff_dict(model))
    assert handoff.agent2_model.species[0].initialization_source == "CURATED"


def test_bounds_default_to_none_when_absent():
    handoff = parse_agent4_handoff(
        build_handoff_dict(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY))
    )
    param = handoff.agent2_model.parameters[0]
    assert param.lower_bound is None
    assert param.upper_bound is None
    assert param.fixed is None


def test_agent3_diagnostics_finding_view_parses():
    diagnostics = {
        "report_id": "r1",
        "agent3_contract_version": "0.1",
        "agent2_model_id": "test-model-1",
        "overall_status": "SUCCESS",
        "diagnostics": [
            {
                "finding_id": "f1",
                "category": "MODEL_STRUCTURE",
                "severity": "WARNING",
                "summary": "something",
                "related_species_ids": ["S1"],
                "related_reaction_ids": [],
                "related_parameter_ids": [],
            }
        ],
        "heuristic_parameter_ids": ["k1"],
        "placeholder_parameter_ids": [],
        "agent2_unresolved_reaction_ids": [],
        "agent2_unresolved_kinetic_law_ids": [],
        "sufficient_for_agent4": True,
    }
    handoff = parse_agent4_handoff(
        build_handoff_dict(minimal_agent2_model_dict(ONE_PARAMETER_ANTIMONY), diagnostics)
    )
    assert len(handoff.agent3_diagnostics.diagnostics) == 1
    assert handoff.agent3_diagnostics.diagnostics[0].related_species_ids == ("S1",)
