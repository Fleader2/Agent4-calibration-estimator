"""The Version 1 calibration-target eligibility policy (``eligibility-v1``).

See ``docs/03_calibration_target_eligibility_policy.md`` for the full rationale. Summary:

* A parameter whose Agent 2 ``source`` is ``HEURISTIC_INITIALIZATION``, ``PLACEHOLDER``, or
  ``DEFAULT`` is calibratable by default -- no explicit authorization needed beyond listing it
  as a ``CalibrationTarget``.
* A parameter whose ``source`` is ``LITERATURE_DERIVED``, ``CURATED``, ``AI_PREDICTED``,
  ``DERIVED_FROM_MACRO_KINETICS``, ``DERIVED_FROM_POOL_CONSERVATION``, or already ``CALIBRATED``
  is protected -- calibratable **only** if the request's own
  ``authorized_provenance_classes`` names that exact source string, or
  ``authorized_parameter_ids`` names that exact parameter id.
* A parameter with ``fixed=True`` (Agent 2's own declared veto) is **never** calibratable,
  regardless of source or any authorization -- this is an absolute rule with no override.
* A species initial condition is calibratable **only** by being listed as an
  ``INITIAL_CONDITION`` ``CalibrationTarget`` -- that listing is itself Version 1's own
  explicit-authorization mechanism. If Agent 2 happens to have recorded a protected
  ``initialization_source`` for that species, the same provenance-class/parameter-id
  authorization check applies as for parameters (the species id substitutes for the parameter
  id in ``authorized_parameter_ids``).
* Bounds: a ``PARAMETER`` target's bounds come from the request's own ``CalibrationTarget
  .bounds`` if given, else from Agent 2's own declared ``lower_bound``/``upper_bound`` if both
  are set, else the target is rejected (no bound is ever invented). An ``INITIAL_CONDITION``
  target's bounds must always come from the request explicitly -- Agent 2 declares no bounds
  for species initial values at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent4.handoff import Agent2Model, Agent4Parameter, Agent4Species
from app.agent4.types import (
    CalibrationBounds,
    CalibrationRequest,
    CalibrationTarget,
    CalibrationTargetType,
)

#: Calibratable without any explicit authorization beyond being listed as a target.
DEFAULT_CALIBRATABLE_PARAMETER_SOURCES = frozenset(
    {"HEURISTIC_INITIALIZATION", "PLACEHOLDER", "DEFAULT"}
)

#: Calibratable only with explicit authorization (``authorized_provenance_classes`` or
#: ``authorized_parameter_ids``) -- evidence-backed, or already the output of a prior
#: calibration run.
PROTECTED_PARAMETER_SOURCES = frozenset(
    {
        "LITERATURE_DERIVED",
        "CURATED",
        "AI_PREDICTED",
        "DERIVED_FROM_MACRO_KINETICS",
        "DERIVED_FROM_POOL_CONSERVATION",
        "CALIBRATED",
    }
)


@dataclass(frozen=True, slots=True)
class EligibilityViolation:
    """One reason a ``CalibrationTarget`` (or the request as a whole) is rejected."""

    target_kind: CalibrationTargetType
    target_id: str
    reason: str


def _parameter_lookup(agent2_model: Agent2Model) -> dict[str, Agent4Parameter]:
    return {p.parameter_id: p for p in agent2_model.parameters}


def _species_lookup(agent2_model: Agent2Model) -> dict[str, Agent4Species]:
    return {s.species_id: s for s in agent2_model.species}


def check_parameter_eligible(parameter: Agent4Parameter, request: CalibrationRequest) -> str | None:
    """``None`` if eligible; otherwise a human-readable rejection reason."""
    if parameter.fixed is True:
        return (
            f"Parameter {parameter.parameter_id!r} has fixed=True in Agent 2's own contract -- "
            "this is an absolute veto on calibration with no override."
        )
    if parameter.source in DEFAULT_CALIBRATABLE_PARAMETER_SOURCES:
        return None
    if parameter.source in PROTECTED_PARAMETER_SOURCES:
        if (
            parameter.source in request.authorized_provenance_classes
            or parameter.parameter_id in request.authorized_parameter_ids
        ):
            return None
        return (
            f"Parameter {parameter.parameter_id!r} has protected source {parameter.source!r} "
            "and is not named in authorized_provenance_classes or authorized_parameter_ids -- "
            "refusing to silently convert an evidence-backed parameter into a fitted one."
        )
    return (
        f"Parameter {parameter.parameter_id!r} has unrecognized source {parameter.source!r} -- "
        "neither a known default-calibratable nor a known protected source; refusing rather "
        "than guessing."
    )


def check_initial_condition_eligible(
    species: Agent4Species, request: CalibrationRequest
) -> str | None:
    """``None`` if eligible; otherwise a human-readable rejection reason.

    Being listed as an ``INITIAL_CONDITION`` target is itself the explicit authorization this
    policy requires -- this function only additionally checks Agent 2's own declared
    ``initialization_source``, when present, against the same protected-class authorization
    parameters use."""
    if species.initialization_source is None:
        return None
    if species.initialization_source in DEFAULT_CALIBRATABLE_PARAMETER_SOURCES:
        return None
    if species.initialization_source in PROTECTED_PARAMETER_SOURCES:
        if (
            species.initialization_source in request.authorized_provenance_classes
            or species.species_id in request.authorized_parameter_ids
        ):
            return None
        return (
            f"Species {species.species_id!r} has protected initialization_source "
            f"{species.initialization_source!r} and is not named in "
            "authorized_provenance_classes or authorized_parameter_ids."
        )
    return (
        f"Species {species.species_id!r} has unrecognized initialization_source "
        f"{species.initialization_source!r} -- refusing rather than guessing."
    )


def resolve_bounds(
    target: CalibrationTarget, agent2_model: Agent2Model
) -> CalibrationBounds | None:
    """The bounds to actually optimize within for ``target``, or ``None`` if none can be
    resolved without inventing one (the caller must then reject the target)."""
    if target.bounds is not None:
        return target.bounds
    if target.target_kind is CalibrationTargetType.PARAMETER:
        parameter = _parameter_lookup(agent2_model).get(target.target_id)
        if (
            parameter is not None
            and parameter.lower_bound is not None
            and parameter.upper_bound is not None
        ):
            return CalibrationBounds(
                lower=float(parameter.lower_bound), upper=float(parameter.upper_bound)
            )
    return None


def validate_request(
    agent2_model: Agent2Model, request: CalibrationRequest
) -> tuple[EligibilityViolation, ...]:
    """Every eligibility violation in ``request`` against ``agent2_model`` -- empty if the
    request is fully valid. Never raises; the caller (``app.agent4.pipeline``) turns a non-empty
    result into ``CalibrationStatus.INVALID_REQUEST``."""
    violations: list[EligibilityViolation] = []
    parameters = _parameter_lookup(agent2_model)
    species = _species_lookup(agent2_model)

    if not request.targets:
        violations.append(
            EligibilityViolation(
                target_kind=CalibrationTargetType.PARAMETER,
                target_id="<none>",
                reason="CalibrationRequest.targets is empty -- nothing to calibrate.",
            )
        )

    seen_ids: set[tuple[CalibrationTargetType, str]] = set()
    for target in request.targets:
        key = (target.target_kind, target.target_id)
        if key in seen_ids:
            violations.append(
                EligibilityViolation(
                    target_kind=target.target_kind,
                    target_id=target.target_id,
                    reason="Duplicate target in the same request.",
                )
            )
            continue
        seen_ids.add(key)

        if target.target_kind is CalibrationTargetType.PARAMETER:
            parameter = parameters.get(target.target_id)
            if parameter is None:
                violations.append(
                    EligibilityViolation(
                        target_kind=target.target_kind,
                        target_id=target.target_id,
                        reason="No parameter with this id exists in the Agent 2 model.",
                    )
                )
                continue
            reason = check_parameter_eligible(parameter, request)
            if reason is not None:
                violations.append(
                    EligibilityViolation(
                        target_kind=target.target_kind, target_id=target.target_id, reason=reason
                    )
                )
                continue
        else:
            species_entry = species.get(target.target_id)
            if species_entry is None:
                violations.append(
                    EligibilityViolation(
                        target_kind=target.target_kind,
                        target_id=target.target_id,
                        reason="No species with this id exists in the Agent 2 model.",
                    )
                )
                continue
            reason = check_initial_condition_eligible(species_entry, request)
            if reason is not None:
                violations.append(
                    EligibilityViolation(
                        target_kind=target.target_kind, target_id=target.target_id, reason=reason
                    )
                )
                continue

        if resolve_bounds(target, agent2_model) is None:
            violations.append(
                EligibilityViolation(
                    target_kind=target.target_kind,
                    target_id=target.target_id,
                    reason=(
                        "No bounds are resolvable for this target: the request supplied none, "
                        "and Agent 2's own declared lower_bound/upper_bound (if any) are not "
                        "both set. Agent 4 never invents a numeric bound."
                    ),
                )
            )

    referenced_species_ids = {obs.target_species_id for obs in request.observations.observations}
    for species_id in referenced_species_ids:
        if species_id not in species:
            violations.append(
                EligibilityViolation(
                    target_kind=CalibrationTargetType.INITIAL_CONDITION,
                    target_id=species_id,
                    reason=(
                        f"Observation references species {species_id!r}, which does not exist "
                        "in the Agent 2 model."
                    ),
                )
            )

    return tuple(violations)


__all__ = [
    "DEFAULT_CALIBRATABLE_PARAMETER_SOURCES",
    "PROTECTED_PARAMETER_SOURCES",
    "EligibilityViolation",
    "check_initial_condition_eligible",
    "check_parameter_eligible",
    "resolve_bounds",
    "validate_request",
]
