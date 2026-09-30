"""The Version 1 calibration workflow, end to end.

``Agent 2 model + Agent 3 diagnostics + observations -> choose authorized targets -> baseline
simulation/objective -> optimize -> validation simulation -> residual/identifiability
diagnostics -> structured Agent 4 report``

Every early-exit branch (``INVALID_REQUEST``, ``INSUFFICIENT_DATA``, ``SIMULATION_FAILED``,
``OPTIMIZATION_FAILED``) still returns a fully-formed ``Agent4Report`` -- never raises for a
genuine calibration outcome, only for a structural contract problem (see
``app.agent4.errors``).
"""

from __future__ import annotations

import uuid

import numpy as np

from app.agent4.antimony_ids import build_species_antimony_id_map
from app.agent4.eligibility import resolve_bounds, validate_request
from app.agent4.errors import Agent4Error
from app.agent4.handoff import Agent2Model, Agent4Handoff
from app.agent4.identifiability import (
    bound_hit_findings,
    correlation_findings,
    degenerate_objective_findings,
    insufficient_observations_finding,
    is_at_bound,
    low_sensitivity_findings,
)
from app.agent4.objective import evaluate_vector, weights_for_observation
from app.agent4.optimizer import run_multistart_optimization
from app.agent4.report import determine_agent5_sufficiency
from app.agent4.types import (
    Agent4Report,
    CalibrationBounds,
    CalibrationRequest,
    CalibrationResult,
    CalibrationStatus,
    CalibrationTarget,
    CalibrationTargetType,
    IdentifiabilityCategory,
    IdentifiabilitySeverity,
    ObservationPartition,
    ParameterEstimate,
    ResidualSummary,
)
from app.agent4.version import AGENT4_CONTRACT_VERSION


def _original_value_for(agent2_model: Agent2Model, target: CalibrationTarget) -> float | None:
    if target.target_kind is CalibrationTargetType.PARAMETER:
        for p in agent2_model.parameters:
            if p.parameter_id == target.target_id:
                return float(p.value) if p.value is not None else None
        return None
    for s in agent2_model.species:
        if s.species_id == target.target_id:
            if s.initial_concentration is not None:
                return float(s.initial_concentration)
            if s.initial_amount is not None:
                return float(s.initial_amount)
            return None
    return None


def _original_source_for(agent2_model: Agent2Model, target: CalibrationTarget) -> str | None:
    if target.target_kind is CalibrationTargetType.PARAMETER:
        for p in agent2_model.parameters:
            if p.parameter_id == target.target_id:
                return p.source
        return None
    for s in agent2_model.species:
        if s.species_id == target.target_id:
            return s.initialization_source
    return None


def _baseline_scalar(
    target: CalibrationTarget, bounds: CalibrationBounds, original_value: float | None
) -> float:
    if target.initial_guess is not None:
        return float(min(max(target.initial_guess, bounds.lower), bounds.upper))
    if original_value is not None:
        return float(min(max(original_value, bounds.lower), bounds.upper))
    return (bounds.lower + bounds.upper) / 2.0


def _empty_result(
    calibration_run_id: str,
    status: CalibrationStatus,
    *,
    objective_before: float | None = None,
    message: str | None = None,
    identifiability_findings: tuple = (),
    n_multistarts: int = 0,
    all_start_objectives: tuple = (),
) -> CalibrationResult:
    return CalibrationResult(
        calibration_run_id=calibration_run_id,
        status=status,
        objective_before=objective_before,
        objective_after=None,
        parameter_estimates=(),
        residuals=(),
        identifiability_findings=identifiability_findings,
        optimizer_message=message,
        n_multistarts=n_multistarts,
        best_start_index=None,
        all_start_objectives=all_start_objectives,
    )


def _build_report(
    report_id: str,
    handoff: Agent4Handoff,
    request: CalibrationRequest,
    calibration: CalibrationResult,
    fixed_parameter_ids: tuple[str, ...],
) -> Agent4Report:
    sufficient, note = determine_agent5_sufficiency(calibration.status)
    return Agent4Report(
        report_id=report_id,
        agent4_contract_version=AGENT4_CONTRACT_VERSION,
        agent2_model_id=handoff.agent2_model.model_id,
        agent3_report_id=handoff.agent3_diagnostics.report_id,
        request_id=request.request_id,
        calibration=calibration,
        fixed_parameters_verified_unchanged=True,
        fixed_parameter_ids_checked=fixed_parameter_ids,
        sufficient_for_agent5=sufficient,
        sufficiency_note=note,
    )


def run_agent4_pipeline(handoff: Agent4Handoff, request: CalibrationRequest) -> Agent4Report:
    """Run the full Version 1 calibration workflow against ``handoff`` and ``request``, and
    return one ``Agent4Report``. Never raises for a genuine calibration outcome -- every such
    outcome is captured in the returned report's own ``CalibrationStatus``."""
    report_id = f"agent4-report-{uuid.uuid4()}"
    calibration_run_id = f"agent4-run-{uuid.uuid4()}"
    agent2_model = handoff.agent2_model
    fixed_parameter_ids = tuple(p.parameter_id for p in agent2_model.parameters if p.fixed is True)

    violations = validate_request(agent2_model, request)
    if violations:
        message = "; ".join(f"{v.target_kind.value}:{v.target_id}: {v.reason}" for v in violations)
        calibration = _empty_result(
            calibration_run_id, CalibrationStatus.INVALID_REQUEST, message=message
        )
        return _build_report(report_id, handoff, request, calibration, fixed_parameter_ids)

    targets = request.targets
    bounds_by_target = [resolve_bounds(t, agent2_model) for t in targets]
    bounds_list = [(b.lower, b.upper) for b in bounds_by_target]
    species_id_map = build_species_antimony_id_map([s.species_id for s in agent2_model.species])

    train_obs = tuple(
        o for o in request.observations.observations if o.partition is ObservationPartition.TRAIN
    )
    validation_obs = tuple(
        o
        for o in request.observations.observations
        if o.partition is ObservationPartition.VALIDATION
    )

    n_train_points = sum(len(o.values) for o in train_obs)
    insufficient = insufficient_observations_finding(n_train_points, len(targets))
    if insufficient is not None:
        calibration = _empty_result(
            calibration_run_id,
            CalibrationStatus.INSUFFICIENT_DATA,
            identifiability_findings=(insufficient,),
        )
        return _build_report(report_id, handoff, request, calibration, fixed_parameter_ids)

    originals = [_original_value_for(agent2_model, t) for t in targets]
    baseline_vector = np.array(
        [
            _baseline_scalar(t, b, orig)
            for t, orig, b in zip(targets, originals, bounds_by_target, strict=True)
        ]
    )

    baseline_eval = evaluate_vector(
        agent2_model, targets, baseline_vector, train_obs, species_id_map
    )
    if not baseline_eval.succeeded:
        calibration = _empty_result(
            calibration_run_id,
            CalibrationStatus.SIMULATION_FAILED,
            message=baseline_eval.message,
        )
        return _build_report(report_id, handoff, request, calibration, fixed_parameter_ids)

    objective_before = baseline_eval.objective

    start_results = run_multistart_optimization(
        agent2_model,
        targets,
        bounds_list,
        baseline_vector,
        train_obs,
        species_id_map,
        request.optimizer,
    )
    all_start_objectives = tuple(r.objective for r in start_results)
    converged = [(i, r) for i, r in enumerate(start_results) if r.converged]
    if not converged:
        calibration = _empty_result(
            calibration_run_id,
            CalibrationStatus.OPTIMIZATION_FAILED,
            objective_before=objective_before,
            message="; ".join(r.message for r in start_results),
            n_multistarts=len(start_results),
            all_start_objectives=all_start_objectives,
        )
        return _build_report(report_id, handoff, request, calibration, fixed_parameter_ids)

    best_index, best_result = min(converged, key=lambda pair: pair[1].objective)
    best_vector = np.array(best_result.final_vector)
    objective_after = best_result.objective

    parameter_estimates = []
    for target, orig_value, bound, value in zip(
        targets, originals, bounds_by_target, best_vector, strict=True
    ):
        at_lower, at_upper = is_at_bound(float(value), bound.lower, bound.upper)
        parameter_estimates.append(
            ParameterEstimate(
                target_kind=target.target_kind,
                target_id=target.target_id,
                original_value=orig_value,
                original_source=_original_source_for(agent2_model, target),
                fitted_value=float(value),
                lower_bound=bound.lower,
                upper_bound=bound.upper,
                at_lower_bound=at_lower,
                at_upper_bound=at_upper,
                provenance_refs=(request.observations.observation_set_id, calibration_run_id),
            )
        )

    for estimate in parameter_estimates:
        if estimate.target_id in fixed_parameter_ids:
            raise Agent4Error(
                f"internal invariant violated: fixed parameter {estimate.target_id!r} was "
                "calibrated -- this should be structurally impossible given app.agent4"
                ".eligibility's own veto; refusing to return a corrupted report."
            )

    full_eval = evaluate_vector(
        agent2_model, targets, best_vector, train_obs + validation_obs, species_id_map
    )
    residual_summaries: list[ResidualSummary] = []
    obs_by_id = {o.observation_id: o for o in request.observations.observations}
    if full_eval.succeeded:
        for rec in full_eval.residuals:
            observation = obs_by_id[rec.observation_id]
            weights = weights_for_observation(observation)
            residuals_arr = np.asarray(rec.residuals)
            residual_summaries.append(
                ResidualSummary(
                    observation_id=rec.observation_id,
                    partition=ObservationPartition(rec.partition),
                    residuals=tuple(rec.residuals),
                    sum_squared_residual=float(np.sum(residuals_arr**2)),
                    weighted_sum_squared_residual=float(np.sum(weights * residuals_arr**2)),
                    n_points=len(rec.residuals),
                )
            )

    findings = list(
        bound_hit_findings(
            [t.target_id for t in targets],
            [e.at_lower_bound for e in parameter_estimates],
            [e.at_upper_bound for e in parameter_estimates],
        )
    )
    findings.extend(
        low_sensitivity_findings(
            agent2_model,
            targets,
            best_vector,
            bounds_list,
            train_obs,
            species_id_map,
            objective_after,
        )
    )
    findings.extend(
        correlation_findings(
            agent2_model, targets, best_vector, bounds_list, train_obs, species_id_map
        )
    )
    findings.extend(degenerate_objective_findings(start_results, bounds_list))

    has_degenerate = any(
        f.category is IdentifiabilityCategory.DEGENERATE_OBJECTIVE for f in findings
    )
    has_warning_or_error = any(
        f.severity in (IdentifiabilitySeverity.WARNING, IdentifiabilitySeverity.ERROR)
        for f in findings
    )
    if has_degenerate:
        status = CalibrationStatus.UNIDENTIFIABLE
    elif has_warning_or_error:
        status = CalibrationStatus.PARTIAL_SUCCESS
    else:
        status = CalibrationStatus.SUCCESS

    calibration = CalibrationResult(
        calibration_run_id=calibration_run_id,
        status=status,
        objective_before=objective_before,
        objective_after=objective_after,
        parameter_estimates=tuple(parameter_estimates),
        residuals=tuple(residual_summaries),
        identifiability_findings=tuple(findings),
        optimizer_message=best_result.message,
        n_multistarts=len(start_results),
        best_start_index=best_index,
        all_start_objectives=all_start_objectives,
    )
    return _build_report(report_id, handoff, request, calibration, fixed_parameter_ids)


__all__ = ["run_agent4_pipeline"]
