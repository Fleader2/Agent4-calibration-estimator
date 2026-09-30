"""The Agent 2/Agent 3 -> Agent 4 input contract (Version 1).

Mirrors this project's own established decoupling pattern (Agent 1 -> Agent 2's
``translate_agent1_view_to_agent2``, Agent 2 -> Agent 3's ``app.agent2.handoff`` boundary):
Agent 4 never imports Agent 2's or Agent 3's Python packages, and never constructs an Agent 2
``ModelSpecification``/``ParameterSpecification`` or an Agent 3 ``Agent3Report`` object
directly. It consumes exactly two plain, JSON-serializable dict shapes, parsed here into this
module's own frozen, validated dataclasses.

**Why two dicts, not one:** the task this repository implements is explicit that "Agent 4 must
consume Agent 3's structured simulation/diagnostic output plus the Agent 2 model/parameter
metadata" -- these are two distinct upstream artifacts with two distinct purposes:

* ``Agent2Model`` -- the actual, compilable ``antimony_text``, plus every structural/provenance
  fact needed to select and bound a calibration target: species/reactions/kinetic laws, and
  every parameter's ``source``/``lower_bound``/``upper_bound``/``fixed``/``uncertainty_text``/
  ``provenance_refs``. This is intentionally a **superset** of the sibling Agent 3 repository's
  own ``Agent2Handoff`` -- Agent 3's Version 1 handoff contract does not carry
  ``lower_bound``/``upper_bound``/``fixed`` (on parameters) or ``initialization_source`` (on
  species), because Agent 3's own job (diagnostics) never needed them. Agent 2's own
  ``ParameterSpecification``/``SpeciesSpecification`` types already carry these fields (confirmed
  by direct inspection of ``agent2-antimony-builder``'s ``app/agent2/types.py`` this session) --
  Agent 4 simply asks for them explicitly in its own handoff contract, since calibration cannot
  select/bound a target without them. This is a new, Agent-4-specific contract, not a
  modification of Agent 2's or Agent 3's existing code or output.
* ``Agent3DiagnosticsView`` -- a lightweight mirror of the parts of Agent 3's own
  ``Agent3Report`` that inform *which* targets are worth considering and what Agent 3 already
  found (e.g. which species Agent 3's own ``missing-initial-conditions`` finding flagged, which
  parameters Agent 3 already classified as heuristic/placeholder). Carried through verbatim,
  never re-derived independently of what Agent 3 actually reported.

Nothing here is a modeling decision or a calibration result -- this module only parses and
validates the shape of what Agent 2 and Agent 3 report about themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.agent4.errors import HandoffContractError


def _require_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HandoffContractError(f"{field} must be a non-empty string, got {value!r}")
    return value


def _optional_str(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HandoffContractError(f"{field} must be a string or null, got {value!r}")
    return value or None


def _optional_decimal(value: object, *, field: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise HandoffContractError(f"{field} must be numeric, got a bool ({value!r})")
    if isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except InvalidOperation as exc:
            raise HandoffContractError(f"{field} is not a valid decimal: {value!r}") from exc
    raise HandoffContractError(f"{field} must be numeric or null, got {value!r}")


def _optional_bool(value: object, *, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise HandoffContractError(f"{field} must be a bool or null, got {value!r}")
    return value


def _str_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) for v in value):
        raise HandoffContractError(f"{field} must be a list of strings, got {value!r}")
    return tuple(value)


def _dict_list(value: object, *, field: str) -> tuple[dict, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, dict) for v in value):
        raise HandoffContractError(f"{field} must be a list of objects, got {value!r}")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class Agent4Compartment:
    compartment_id: str
    name: str
    initial_volume: Decimal | None
    volume_unit: str | None
    constant: bool | None


@dataclass(frozen=True, slots=True)
class Agent4Species:
    species_id: str
    name: str
    compartment_id: str
    source_compound_id: str | None
    source_enzyme_state_id: str | None
    initial_amount: Decimal | None
    initial_concentration: Decimal | None
    constant: bool | None
    boundary_condition: bool | None
    #: Agent 2's own ``SpeciesSpecification.initialization_source`` (a ``ParameterSource``
    #: value, or null when Agent 2 never set one) -- present on Agent 2's real type but not
    #: carried by Agent 3's own Version 1 handoff contract (see this module's own docstring).
    initialization_source: str | None


@dataclass(frozen=True, slots=True)
class Agent4Participant:
    species_id: str
    role: str
    stoichiometry: Decimal


@dataclass(frozen=True, slots=True)
class Agent4Reaction:
    reaction_id: str
    name: str | None
    reversible: bool | None
    participants: tuple[Agent4Participant, ...]


@dataclass(frozen=True, slots=True)
class Agent4KineticLaw:
    kinetic_law_id: str
    reaction_id: str
    law_type: str
    expression: str | None
    parameter_ids: tuple[str, ...]
    species_ids: tuple[str, ...]
    enzyme_state_id: str | None
    protein_id: str | None
    complex_id: str | None
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Agent4Parameter:
    parameter_id: str
    name: str
    source: str
    value: Decimal | None
    unit: str | None
    reaction_id: str | None
    kinetic_law_assignment_id: str | None
    uncertainty_text: str | None
    provenance_refs: tuple[str, ...]
    #: Agent 2's own ``ParameterSpecification.lower_bound``/``.upper_bound``/``.fixed`` --
    #: present on Agent 2's real type but not carried by Agent 3's own Version 1 handoff
    #: contract (see this module's own docstring). ``fixed=True`` is an absolute veto on
    #: calibration (see ``app.agent4.eligibility``), never overridable by any request.
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    fixed: bool | None


@dataclass(frozen=True, slots=True)
class Agent4ModelAssumption:
    assumption_id: str
    category: str
    statement: str
    related_entity_ids: tuple[str, ...]
    reason_code: str | None


@dataclass(frozen=True, slots=True)
class Agent2Model:
    """The Agent 2 half of the Agent 4 handoff: the compilable model plus every
    structural/provenance fact needed to select and bound a calibration target."""

    contract_version: str
    model_id: str
    network_id: str
    antimony_text: str
    antimony_generator_version: str
    readiness: str
    unresolved_reaction_ids: tuple[str, ...]
    unresolved_kinetic_law_ids: tuple[str, ...]
    compartments: tuple[Agent4Compartment, ...]
    species: tuple[Agent4Species, ...]
    reactions: tuple[Agent4Reaction, ...]
    kinetic_laws: tuple[Agent4KineticLaw, ...]
    parameters: tuple[Agent4Parameter, ...]
    model_assumptions: tuple[Agent4ModelAssumption, ...]


@dataclass(frozen=True, slots=True)
class Agent4DiagnosticFindingView:
    """A lightweight mirror of one Agent 3 ``DiagnosticFinding`` -- carried through verbatim,
    never re-derived. Agent 4 reads these (e.g. to cross-check which species Agent 3's own
    ``missing-initial-conditions`` finding flagged) but never recomputes Agent 3's own
    diagnostic logic itself."""

    finding_id: str
    category: str
    severity: str
    summary: str
    related_species_ids: tuple[str, ...]
    related_reaction_ids: tuple[str, ...]
    related_parameter_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Agent3DiagnosticsView:
    """The Agent 3 half of the Agent 4 handoff: a lightweight mirror of the parts of
    ``Agent3Report`` relevant to choosing and interpreting calibration targets."""

    report_id: str
    agent3_contract_version: str
    agent2_model_id: str
    overall_status: str
    diagnostics: tuple[Agent4DiagnosticFindingView, ...]
    heuristic_parameter_ids: tuple[str, ...]
    placeholder_parameter_ids: tuple[str, ...]
    agent2_unresolved_reaction_ids: tuple[str, ...]
    agent2_unresolved_kinetic_law_ids: tuple[str, ...]
    sufficient_for_agent4: bool


@dataclass(frozen=True, slots=True)
class Agent4Handoff:
    """One complete, self-contained Agent 2/Agent 3 -> Agent 4 handoff."""

    agent2_model: Agent2Model
    agent3_diagnostics: Agent3DiagnosticsView


def _parse_agent2_model(data: dict) -> Agent2Model:
    compartments = tuple(
        Agent4Compartment(
            compartment_id=_require_str(
                c.get("compartment_id"), field="agent2_model.compartments[].compartment_id"
            ),
            name=_require_str(c.get("name"), field="agent2_model.compartments[].name"),
            initial_volume=_optional_decimal(
                c.get("initial_volume"), field="agent2_model.compartments[].initial_volume"
            ),
            volume_unit=_optional_str(
                c.get("volume_unit"), field="agent2_model.compartments[].volume_unit"
            ),
            constant=c.get("constant"),
        )
        for c in _dict_list(data.get("compartments"), field="agent2_model.compartments")
    )

    species = tuple(
        Agent4Species(
            species_id=_require_str(s.get("species_id"), field="agent2_model.species[].species_id"),
            name=_require_str(s.get("name"), field="agent2_model.species[].name"),
            compartment_id=_require_str(
                s.get("compartment_id"), field="agent2_model.species[].compartment_id"
            ),
            source_compound_id=_optional_str(
                s.get("source_compound_id"), field="agent2_model.species[].source_compound_id"
            ),
            source_enzyme_state_id=_optional_str(
                s.get("source_enzyme_state_id"),
                field="agent2_model.species[].source_enzyme_state_id",
            ),
            initial_amount=_optional_decimal(
                s.get("initial_amount"), field="agent2_model.species[].initial_amount"
            ),
            initial_concentration=_optional_decimal(
                s.get("initial_concentration"),
                field="agent2_model.species[].initial_concentration",
            ),
            constant=s.get("constant"),
            boundary_condition=s.get("boundary_condition"),
            initialization_source=_optional_str(
                s.get("initialization_source"),
                field="agent2_model.species[].initialization_source",
            ),
        )
        for s in _dict_list(data.get("species"), field="agent2_model.species")
    )

    reactions = tuple(
        Agent4Reaction(
            reaction_id=_require_str(
                r.get("reaction_id"), field="agent2_model.reactions[].reaction_id"
            ),
            name=_optional_str(r.get("name"), field="agent2_model.reactions[].name"),
            reversible=r.get("reversible"),
            participants=tuple(
                Agent4Participant(
                    species_id=_require_str(
                        p.get("species_id"),
                        field="agent2_model.reactions[].participants[].species_id",
                    ),
                    role=_require_str(
                        p.get("role"), field="agent2_model.reactions[].participants[].role"
                    ),
                    stoichiometry=_optional_decimal(
                        p.get("stoichiometry"),
                        field="agent2_model.reactions[].participants[].stoichiometry",
                    )
                    or Decimal(1),
                )
                for p in _dict_list(
                    r.get("participants"), field="agent2_model.reactions[].participants"
                )
            ),
        )
        for r in _dict_list(data.get("reactions"), field="agent2_model.reactions")
    )

    kinetic_laws = tuple(
        Agent4KineticLaw(
            kinetic_law_id=_require_str(
                k.get("kinetic_law_id"), field="agent2_model.kinetic_laws[].kinetic_law_id"
            ),
            reaction_id=_require_str(
                k.get("reaction_id"), field="agent2_model.kinetic_laws[].reaction_id"
            ),
            law_type=_require_str(k.get("law_type"), field="agent2_model.kinetic_laws[].law_type"),
            expression=_optional_str(
                k.get("expression"), field="agent2_model.kinetic_laws[].expression"
            ),
            parameter_ids=_str_tuple(
                k.get("parameter_ids"), field="agent2_model.kinetic_laws[].parameter_ids"
            ),
            species_ids=_str_tuple(
                k.get("species_ids"), field="agent2_model.kinetic_laws[].species_ids"
            ),
            enzyme_state_id=_optional_str(
                k.get("enzyme_state_id"), field="agent2_model.kinetic_laws[].enzyme_state_id"
            ),
            protein_id=_optional_str(
                k.get("protein_id"), field="agent2_model.kinetic_laws[].protein_id"
            ),
            complex_id=_optional_str(
                k.get("complex_id"), field="agent2_model.kinetic_laws[].complex_id"
            ),
            assumptions=_str_tuple(
                k.get("assumptions"), field="agent2_model.kinetic_laws[].assumptions"
            ),
        )
        for k in _dict_list(data.get("kinetic_laws"), field="agent2_model.kinetic_laws")
    )

    parameters = tuple(
        Agent4Parameter(
            parameter_id=_require_str(
                p.get("parameter_id"), field="agent2_model.parameters[].parameter_id"
            ),
            name=_require_str(p.get("name"), field="agent2_model.parameters[].name"),
            source=_require_str(p.get("source"), field="agent2_model.parameters[].source"),
            value=_optional_decimal(p.get("value"), field="agent2_model.parameters[].value"),
            unit=_optional_str(p.get("unit"), field="agent2_model.parameters[].unit"),
            reaction_id=_optional_str(
                p.get("reaction_id"), field="agent2_model.parameters[].reaction_id"
            ),
            kinetic_law_assignment_id=_optional_str(
                p.get("kinetic_law_assignment_id"),
                field="agent2_model.parameters[].kinetic_law_assignment_id",
            ),
            uncertainty_text=_optional_str(
                p.get("uncertainty_text"), field="agent2_model.parameters[].uncertainty_text"
            ),
            provenance_refs=_str_tuple(
                p.get("provenance_refs"), field="agent2_model.parameters[].provenance_refs"
            ),
            lower_bound=_optional_decimal(
                p.get("lower_bound"), field="agent2_model.parameters[].lower_bound"
            ),
            upper_bound=_optional_decimal(
                p.get("upper_bound"), field="agent2_model.parameters[].upper_bound"
            ),
            fixed=_optional_bool(p.get("fixed"), field="agent2_model.parameters[].fixed"),
        )
        for p in _dict_list(data.get("parameters"), field="agent2_model.parameters")
    )

    model_assumptions = tuple(
        Agent4ModelAssumption(
            assumption_id=_require_str(
                a.get("assumption_id"), field="agent2_model.model_assumptions[].assumption_id"
            ),
            category=_require_str(
                a.get("category"), field="agent2_model.model_assumptions[].category"
            ),
            statement=_require_str(
                a.get("statement"), field="agent2_model.model_assumptions[].statement"
            ),
            related_entity_ids=_str_tuple(
                a.get("related_entity_ids"),
                field="agent2_model.model_assumptions[].related_entity_ids",
            ),
            reason_code=_optional_str(
                a.get("reason_code"), field="agent2_model.model_assumptions[].reason_code"
            ),
        )
        for a in _dict_list(data.get("model_assumptions"), field="agent2_model.model_assumptions")
    )

    return Agent2Model(
        contract_version=_require_str(
            data.get("contract_version"), field="agent2_model.contract_version"
        ),
        model_id=_require_str(data.get("model_id"), field="agent2_model.model_id"),
        network_id=_require_str(data.get("network_id"), field="agent2_model.network_id"),
        antimony_text=_require_str(data.get("antimony_text"), field="agent2_model.antimony_text"),
        antimony_generator_version=_require_str(
            data.get("antimony_generator_version"),
            field="agent2_model.antimony_generator_version",
        ),
        readiness=_require_str(data.get("readiness"), field="agent2_model.readiness"),
        unresolved_reaction_ids=_str_tuple(
            data.get("unresolved_reaction_ids"), field="agent2_model.unresolved_reaction_ids"
        ),
        unresolved_kinetic_law_ids=_str_tuple(
            data.get("unresolved_kinetic_law_ids"),
            field="agent2_model.unresolved_kinetic_law_ids",
        ),
        compartments=compartments,
        species=species,
        reactions=reactions,
        kinetic_laws=kinetic_laws,
        parameters=parameters,
        model_assumptions=model_assumptions,
    )


def _parse_agent3_diagnostics(data: dict) -> Agent3DiagnosticsView:
    diagnostics = tuple(
        Agent4DiagnosticFindingView(
            finding_id=_require_str(
                f.get("finding_id"), field="agent3_diagnostics.diagnostics[].finding_id"
            ),
            category=_require_str(
                f.get("category"), field="agent3_diagnostics.diagnostics[].category"
            ),
            severity=_require_str(
                f.get("severity"), field="agent3_diagnostics.diagnostics[].severity"
            ),
            summary=_require_str(
                f.get("summary"), field="agent3_diagnostics.diagnostics[].summary"
            ),
            related_species_ids=_str_tuple(
                f.get("related_species_ids"),
                field="agent3_diagnostics.diagnostics[].related_species_ids",
            ),
            related_reaction_ids=_str_tuple(
                f.get("related_reaction_ids"),
                field="agent3_diagnostics.diagnostics[].related_reaction_ids",
            ),
            related_parameter_ids=_str_tuple(
                f.get("related_parameter_ids"),
                field="agent3_diagnostics.diagnostics[].related_parameter_ids",
            ),
        )
        for f in _dict_list(data.get("diagnostics"), field="agent3_diagnostics.diagnostics")
    )
    return Agent3DiagnosticsView(
        report_id=_require_str(data.get("report_id"), field="agent3_diagnostics.report_id"),
        agent3_contract_version=_require_str(
            data.get("agent3_contract_version"),
            field="agent3_diagnostics.agent3_contract_version",
        ),
        agent2_model_id=_require_str(
            data.get("agent2_model_id"), field="agent3_diagnostics.agent2_model_id"
        ),
        overall_status=_require_str(
            data.get("overall_status"), field="agent3_diagnostics.overall_status"
        ),
        diagnostics=diagnostics,
        heuristic_parameter_ids=_str_tuple(
            data.get("heuristic_parameter_ids"),
            field="agent3_diagnostics.heuristic_parameter_ids",
        ),
        placeholder_parameter_ids=_str_tuple(
            data.get("placeholder_parameter_ids"),
            field="agent3_diagnostics.placeholder_parameter_ids",
        ),
        agent2_unresolved_reaction_ids=_str_tuple(
            data.get("agent2_unresolved_reaction_ids"),
            field="agent3_diagnostics.agent2_unresolved_reaction_ids",
        ),
        agent2_unresolved_kinetic_law_ids=_str_tuple(
            data.get("agent2_unresolved_kinetic_law_ids"),
            field="agent3_diagnostics.agent2_unresolved_kinetic_law_ids",
        ),
        sufficient_for_agent4=bool(data.get("sufficient_for_agent4", False)),
    )


def parse_agent4_handoff(data: dict) -> Agent4Handoff:
    """Parse and validate one plain dict (with ``agent2_model`` and ``agent3_diagnostics`` keys)
    into an ``Agent4Handoff``.

    Raises ``HandoffContractError`` for any structurally malformed input. Never fills in a
    missing value with a guessed default -- exactly mirrors ``app.agent3.handoff
    .parse_agent2_handoff``'s own discipline on the sibling repository's identical boundary.
    """
    if not isinstance(data, dict):
        raise HandoffContractError(f"Agent 4 handoff must be a dict, got {type(data)!r}")
    agent2_model_data = data.get("agent2_model")
    if not isinstance(agent2_model_data, dict):
        raise HandoffContractError(f"agent2_model must be a dict, got {type(agent2_model_data)!r}")
    agent3_diagnostics_data = data.get("agent3_diagnostics")
    if not isinstance(agent3_diagnostics_data, dict):
        raise HandoffContractError(
            f"agent3_diagnostics must be a dict, got {type(agent3_diagnostics_data)!r}"
        )
    return Agent4Handoff(
        agent2_model=_parse_agent2_model(agent2_model_data),
        agent3_diagnostics=_parse_agent3_diagnostics(agent3_diagnostics_data),
    )


def handoff_contract_version_note(agent2_model: Agent2Model) -> str | None:
    """``None`` if nothing to disclose; otherwise a human-readable note -- never raised as an
    error (see ``Agent4Handoff``'s own docstring). Compares the *Agent 2* contract version
    embedded in the handoff against no fixed expectation (Agent 4 has no single upstream
    contract version it demands) -- this is a placeholder for future version reconciliation,
    always returning ``None`` in Version 1, consistent with ``AGENT4_HANDOFF_CONTRACT_VERSION``
    only describing Agent 4's *own* handoff shape, not Agent 2's or Agent 3's."""
    del agent2_model
    return None


__all__ = [
    "Agent2Model",
    "Agent3DiagnosticsView",
    "Agent4Compartment",
    "Agent4DiagnosticFindingView",
    "Agent4Handoff",
    "Agent4KineticLaw",
    "Agent4ModelAssumption",
    "Agent4Parameter",
    "Agent4Participant",
    "Agent4Reaction",
    "Agent4Species",
    "handoff_contract_version_note",
    "parse_agent4_handoff",
]
