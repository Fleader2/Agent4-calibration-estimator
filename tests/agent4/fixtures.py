"""Shared, hand-crafted fixtures for Agent 4's own test suite.

Every synthetic model here is small and clearly labeled as synthetic -- per the task's own
explicit instruction, Version 1's calibration-recovery tests are proven against known-ground-
truth synthetic models, never against fabricated yeast biology. The real sce00061 fixture
(``load_real_sce00061_handoff``) is used only to prove Agent 4 can *parse* the real model and
correctly identify calibration-eligible targets -- never to pretend a real biological fit
happened without real observation data.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent4.handoff import parse_agent4_handoff
from app.agent4.loader import load_antimony_model

ONE_PARAMETER_TRUE_K = 0.7

#: A single first-order decay: dS1/dt = -k1*S1. One identifiable parameter.
ONE_PARAMETER_ANTIMONY = f"""
model decay
  compartment C = 1
  species s_S1 in C
  s_S1 = 10
  J1: s_S1 -> ;
  J1 = p_k1 * s_S1
  p_k1 = {ONE_PARAMETER_TRUE_K}
end
"""

TWO_PARAMETER_TRUE_K1 = 0.5
TWO_PARAMETER_TRUE_K2 = 0.2

#: A two-step irreversible chain S1 -> S2 -> S3. k1 != k2, and every species is observable, so
#: both rate constants are structurally identifiable from the transient shape alone.
TWO_PARAMETER_ANTIMONY = f"""
model chain
  compartment C = 1
  species s_S1 in C
  species s_S2 in C
  species s_S3 in C
  s_S1 = 10
  s_S2 = 0
  s_S3 = 0
  J1: s_S1 -> s_S2;
  J1 = p_k1 * s_S1
  J2: s_S2 -> s_S3;
  J2 = p_k2 * s_S2
  p_k1 = {TWO_PARAMETER_TRUE_K1}
  p_k2 = {TWO_PARAMETER_TRUE_K2}
end
"""

UNIDENTIFIABLE_TRUE_PRODUCT = 0.3

#: S1 -> S2 at rate (k1*k2)*S1 -- only the PRODUCT k1*k2 is identifiable from any observation of
#: S1(t)/S2(t); individually, k1 and k2 are not (an infinite ridge of equally-good pairs with
#: the same product fits the data equally well). Deliberately constructed non-identifiable pair.
UNIDENTIFIABLE_ANTIMONY = f"""
model unidentifiable
  compartment C = 1
  species s_S1 in C
  species s_S2 in C
  s_S1 = 10
  s_S2 = 0
  J1: s_S1 -> s_S2;
  J1 = p_k1 * p_k2 * s_S1
  p_k1 = {UNIDENTIFIABLE_TRUE_PRODUCT}
  p_k2 = 1.0
end
"""

MISSING_IC_TRUE_VALUE = 5.0
MISSING_IC_KNOWN_K = 0.3

#: Simple decay with a KNOWN, fixed rate constant (CURATED, never a target) and an initial
#: concentration Agent 2 never declared (None/None) -- the only calibration target is the
#: initial condition itself.
MISSING_IC_ANTIMONY = f"""
model missing_ic
  compartment C = 1
  species s_S1 in C
  s_S1 = 0
  J1: s_S1 -> ;
  J1 = p_k1 * s_S1
  p_k1 = {MISSING_IC_KNOWN_K}
end
"""


def minimal_agent2_model_dict(antimony_text: str, **overrides) -> dict:
    """The smallest valid ``agent2_model`` dict -- callers override ``species``/``parameters``/
    etc. to describe the specific synthetic model they built."""
    base = {
        "contract_version": "agent2-downstream-v1",
        "model_id": "test-model-1",
        "network_id": "test-network-1",
        "antimony_text": antimony_text,
        "antimony_generator_version": "test-generator-v0",
        "readiness": "EXECUTABLE",
        "unresolved_reaction_ids": [],
        "unresolved_kinetic_law_ids": [],
        "compartments": [
            {
                "compartment_id": "C",
                "name": "C",
                "initial_volume": "1",
                "volume_unit": "L",
                "constant": True,
            }
        ],
        "species": [
            {
                "species_id": "S1",
                "name": "S1",
                "compartment_id": "C",
                "source_compound_id": None,
                "source_enzyme_state_id": None,
                "initial_amount": None,
                "initial_concentration": "10",
                "constant": False,
                "boundary_condition": False,
                "initialization_source": None,
            }
        ],
        "reactions": [
            {
                "reaction_id": "reaction-1",
                "name": "decay",
                "reversible": False,
                "participants": [{"species_id": "S1", "role": "REACTANT", "stoichiometry": "1"}],
            }
        ],
        "kinetic_laws": [
            {
                "kinetic_law_id": "reaction-1::kinetic-law::none",
                "reaction_id": "reaction-1",
                "law_type": "MASS_ACTION",
                "expression": "k1 * S1",
                "parameter_ids": ["k1"],
                "species_ids": ["S1"],
                "enzyme_state_id": None,
                "protein_id": None,
                "complex_id": None,
                "assumptions": [],
            }
        ],
        "parameters": [
            {
                "parameter_id": "k1",
                "name": "k1",
                "source": "HEURISTIC_INITIALIZATION",
                "value": "0.1",
                "unit": "per_sec",
                "reaction_id": "reaction-1",
                "kinetic_law_assignment_id": "reaction-1::kinetic-law::none",
                "uncertainty_text": "Heuristic default; requires calibration.",
                "provenance_refs": [],
                "lower_bound": None,
                "upper_bound": None,
                "fixed": None,
            }
        ],
        "model_assumptions": [],
    }
    base.update(overrides)
    return base


def minimal_agent3_diagnostics_dict(**overrides) -> dict:
    base = {
        "report_id": "agent3-report-test",
        "agent3_contract_version": "0.1",
        "agent2_model_id": "test-model-1",
        "overall_status": "SUCCESS",
        "diagnostics": [],
        "heuristic_parameter_ids": ["k1"],
        "placeholder_parameter_ids": [],
        "agent2_unresolved_reaction_ids": [],
        "agent2_unresolved_kinetic_law_ids": [],
        "sufficient_for_agent4": True,
    }
    base.update(overrides)
    return base


def build_handoff_dict(agent2_model: dict, agent3_diagnostics: dict | None = None) -> dict:
    return {
        "agent2_model": agent2_model,
        "agent3_diagnostics": agent3_diagnostics or minimal_agent3_diagnostics_dict(),
    }


def parsed_handoff(agent2_model: dict, agent3_diagnostics: dict | None = None):
    return parse_agent4_handoff(build_handoff_dict(agent2_model, agent3_diagnostics))


def simulate_truth(antimony_text: str, species_id: str, end_time: float, num_points: int):
    """Simulate ``antimony_text`` as-is (its own declared "true" parameter values) and return
    ``(time_points, values)`` for ``species_id`` -- used to build clearly-labeled synthetic
    observations from a known ground truth."""
    load_result = load_antimony_model(antimony_text)
    assert load_result.succeeded, load_result.message
    rr = load_result.roadrunner_instance
    sim = rr.simulate(0.0, end_time, num_points)
    colnames = [c.strip("[]") for c in sim.colnames]
    col_index = colnames.index(species_id)
    time_points = tuple(float(t) for t in sim[:, 0])
    values = tuple(float(v) for v in sim[:, col_index])
    return time_points, values


def load_real_sce00061_handoff() -> dict:
    """The real, saved Agent 2/Agent 3 handoff for the current ``sce00061`` model -- built once,
    in this session, by merging the sibling Agent 3 repository's own real handoff fixture with
    a real extract of Agent 2's ``lower_bound``/``upper_bound``/``fixed``/
    ``initialization_source`` fields (absent from Agent 3's own Version 1 handoff, since Agent
    3's own job never needed them) and a fresh, real ``Agent3Report`` run. Used only for this
    package's own bounded real-integration test -- never for a fabricated biological fit."""
    fixture_path = (
        Path(__file__).resolve().parent.parent / "fixtures" / "sce00061_agent4_handoff.json"
    )
    with open(fixture_path) as f:
        return json.load(f)


__all__ = [
    "MISSING_IC_ANTIMONY",
    "MISSING_IC_KNOWN_K",
    "MISSING_IC_TRUE_VALUE",
    "ONE_PARAMETER_ANTIMONY",
    "ONE_PARAMETER_TRUE_K",
    "TWO_PARAMETER_ANTIMONY",
    "TWO_PARAMETER_TRUE_K1",
    "TWO_PARAMETER_TRUE_K2",
    "UNIDENTIFIABLE_ANTIMONY",
    "UNIDENTIFIABLE_TRUE_PRODUCT",
    "build_handoff_dict",
    "load_real_sce00061_handoff",
    "minimal_agent2_model_dict",
    "minimal_agent3_diagnostics_dict",
    "parsed_handoff",
    "simulate_truth",
]
