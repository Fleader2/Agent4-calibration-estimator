"""Version 1 practical-identifiability diagnostics (``identifiability-v1``).

Every check here is a **local, approximate, practical** heuristic near the fitted optimum or
across the multi-start results already computed -- none of them is, or claims to be, a formal
structural-identifiability proof (e.g. differential-algebra-based observability analysis).
See ``docs/05_identifiability_diagnostics.md`` for exactly what each category does and does not
establish.
"""

from __future__ import annotations

import numpy as np

from app.agent4.handoff import Agent2Model
from app.agent4.objective import evaluate_vector
from app.agent4.optimizer import StartResult
from app.agent4.types import (
    CalibrationTarget,
    IdentifiabilityCategory,
    IdentifiabilityFinding,
    IdentifiabilitySeverity,
    Observation,
)

#: A fitted value within this fraction of the bound's own range counts as "at the bound."
_BOUND_HIT_RELATIVE_TOLERANCE = 0.01

#: Relative finite-difference step (of each target's own bound range) used for local
#: sensitivity/correlation checks.
_SENSITIVITY_RELATIVE_STEP = 0.01

#: Below this relative change in objective for a ``_SENSITIVITY_RELATIVE_STEP`` nudge, a target
#: is flagged as weakly sensitive.
_LOW_SENSITIVITY_RELATIVE_THRESHOLD = 1e-6

#: An absolute floor on top of the relative threshold above -- protects against a near-zero
#: (e.g. noiseless, near-perfectly-fit) baseline objective, where the simulation/interpolation
#: pipeline's own numerical noise floor (empirically observed around 1e-11 for the small models
#: this Version 1 policy targets) would otherwise dominate a purely relative comparison and
#: produce a spurious "sensitive" classification for a target with zero true effect.
_LOW_SENSITIVITY_ABSOLUTE_FLOOR = 1e-8

#: Above this absolute correlation coefficient, a fitted-parameter pair is flagged as highly
#: correlated (practically non-identifiable as a pair).
_HIGH_CORRELATION_THRESHOLD = 0.95

#: Two converged multi-starts are treated as "the same fit" if their parameter vectors are
#: closer than this normalized (by bound range) Euclidean distance -- above it, with a near-
#: identical objective, they are flagged as a degenerate/flat-objective pair.
_DEGENERATE_PARAMETER_DISTANCE_THRESHOLD = 0.2
_DEGENERATE_OBJECTIVE_RELATIVE_THRESHOLD = 1e-3


def is_at_bound(value: float, lower: float, upper: float) -> tuple[bool, bool]:
    """``(at_lower_bound, at_upper_bound)`` -- single source of truth shared by
    ``app.agent4.pipeline`` (to populate ``ParameterEstimate``) and ``bound_hit_findings`` below,
    so a value is never classified differently in the two places."""
    span = upper - lower
    if span <= 0:
        return (value <= lower, value >= upper)
    tol = span * _BOUND_HIT_RELATIVE_TOLERANCE
    return (value - lower <= tol, upper - value <= tol)


def bound_hit_findings(
    target_ids: list[str],
    at_lower: list[bool],
    at_upper: list[bool],
) -> tuple[IdentifiabilityFinding, ...]:
    findings = []
    for target_id, lower_hit, upper_hit in zip(target_ids, at_lower, at_upper, strict=True):
        if lower_hit or upper_hit:
            which = "lower" if lower_hit else "upper"
            findings.append(
                IdentifiabilityFinding(
                    finding_id=f"bound-hit:{target_id}",
                    category=IdentifiabilityCategory.BOUND_HIT,
                    severity=IdentifiabilitySeverity.WARNING,
                    summary=f"Target {target_id!r} converged at its {which} bound.",
                    explanation=(
                        "A fitted value sitting at its own declared bound usually means the "
                        "bound itself is too tight, or the data does not actually constrain "
                        "this parameter away from that edge -- this is not a reliable point "
                        "estimate as reported."
                    ),
                    related_target_ids=(target_id,),
                )
            )
    return tuple(findings)


def low_sensitivity_findings(
    agent2_model: Agent2Model,
    targets: tuple[CalibrationTarget, ...],
    best_vector: np.ndarray,
    bounds: list[tuple[float, float]],
    train_observations: tuple[Observation, ...],
    species_id_map: dict[str, str],
    baseline_objective: float,
) -> tuple[IdentifiabilityFinding, ...]:
    findings = []
    #: Combines a relative criterion (scales with how large the baseline objective already is)
    #: with an absolute floor (protects against a near-zero baseline, where the simulation/
    #: interpolation pipeline's own numerical noise would otherwise dominate a purely relative
    #: comparison) -- see ``_LOW_SENSITIVITY_ABSOLUTE_FLOOR``'s own docstring.
    absolute_threshold = max(
        _LOW_SENSITIVITY_ABSOLUTE_FLOOR, _LOW_SENSITIVITY_RELATIVE_THRESHOLD * baseline_objective
    )
    for i, target in enumerate(targets):
        lo, hi = bounds[i]
        step = max((hi - lo) * _SENSITIVITY_RELATIVE_STEP, 1e-12)
        nudged = best_vector.copy()
        nudged[i] = min(nudged[i] + step, hi)
        nudged_objective = evaluate_vector(
            agent2_model, targets, nudged, train_observations, species_id_map
        ).objective
        absolute_change = abs(nudged_objective - baseline_objective)
        if absolute_change < absolute_threshold:
            findings.append(
                IdentifiabilityFinding(
                    finding_id=f"low-sensitivity:{target.target_id}",
                    category=IdentifiabilityCategory.LOW_SENSITIVITY,
                    severity=IdentifiabilitySeverity.WARNING,
                    summary=(
                        f"Objective changes by only {absolute_change:.2e} (absolute) for a "
                        f"{_SENSITIVITY_RELATIVE_STEP:.0%} nudge to {target.target_id!r}."
                    ),
                    explanation=(
                        "A small local perturbation to this target barely moves the objective -- "
                        "the supplied observations carry little information about its value near "
                        "the fitted optimum."
                    ),
                    related_target_ids=(target.target_id,),
                )
            )
    return tuple(findings)


def _finite_difference_jacobian(
    agent2_model: Agent2Model,
    targets: tuple[CalibrationTarget, ...],
    best_vector: np.ndarray,
    bounds: list[tuple[float, float]],
    train_observations: tuple[Observation, ...],
    species_id_map: dict[str, str],
) -> np.ndarray | None:
    """Central-difference Jacobian of the full residual vector w.r.t. each target, evaluated at
    ``best_vector`` -- used only for the local correlation approximation below. ``None`` if any
    evaluation along the way fails to simulate."""
    base_eval = evaluate_vector(
        agent2_model, targets, best_vector, train_observations, species_id_map
    )
    if not base_eval.succeeded:
        return None
    base_residuals = np.concatenate([np.asarray(r.residuals) for r in base_eval.residuals])

    columns = []
    for i in range(len(targets)):
        lo, hi = bounds[i]
        step = max((hi - lo) * _SENSITIVITY_RELATIVE_STEP, 1e-12)
        plus = best_vector.copy()
        plus[i] = min(plus[i] + step, hi)
        minus = best_vector.copy()
        minus[i] = max(minus[i] - step, lo)

        plus_eval = evaluate_vector(agent2_model, targets, plus, train_observations, species_id_map)
        minus_eval = evaluate_vector(
            agent2_model, targets, minus, train_observations, species_id_map
        )
        if not plus_eval.succeeded or not minus_eval.succeeded:
            return None
        plus_residuals = np.concatenate([np.asarray(r.residuals) for r in plus_eval.residuals])
        minus_residuals = np.concatenate([np.asarray(r.residuals) for r in minus_eval.residuals])
        if (
            plus_residuals.shape != base_residuals.shape
            or minus_residuals.shape != base_residuals.shape
        ):
            return None
        columns.append((plus_residuals - minus_residuals) / (plus[i] - minus[i]))
    return np.column_stack(columns)


def correlation_findings(
    agent2_model: Agent2Model,
    targets: tuple[CalibrationTarget, ...],
    best_vector: np.ndarray,
    bounds: list[tuple[float, float]],
    train_observations: tuple[Observation, ...],
    species_id_map: dict[str, str],
) -> tuple[IdentifiabilityFinding, ...]:
    if len(targets) < 2:
        return ()
    jacobian = _finite_difference_jacobian(
        agent2_model, targets, best_vector, bounds, train_observations, species_id_map
    )
    if jacobian is None:
        return ()
    fisher_information = jacobian.T @ jacobian
    try:
        covariance = np.linalg.pinv(fisher_information)
    except np.linalg.LinAlgError:  # pragma: no cover - pinv itself rarely raises
        return ()

    diag = np.sqrt(np.clip(np.diag(covariance), 1e-300, None))
    findings = []
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            denom = diag[i] * diag[j]
            if denom <= 0:
                continue
            corr = covariance[i, j] / denom
            if abs(corr) >= _HIGH_CORRELATION_THRESHOLD:
                findings.append(
                    IdentifiabilityFinding(
                        finding_id=f"high-correlation:{targets[i].target_id}:{targets[j].target_id}",
                        category=IdentifiabilityCategory.HIGH_CORRELATION,
                        severity=IdentifiabilitySeverity.ERROR,
                        summary=(
                            f"Targets {targets[i].target_id!r} and {targets[j].target_id!r} "
                            f"have an estimated local correlation of {corr:.3f}."
                        ),
                        explanation=(
                            "A local linearization (finite-difference Jacobian of the residual "
                            "vector at the fitted optimum) suggests these two targets are "
                            "practically indistinguishable from the supplied observations -- "
                            "only some combination of them, not their individual values, is "
                            "actually constrained by the data. This is a local approximation, "
                            "never a formal structural-identifiability proof."
                        ),
                        related_target_ids=(targets[i].target_id, targets[j].target_id),
                    )
                )
    return tuple(findings)


def degenerate_objective_findings(
    start_results: list[StartResult], bounds: list[tuple[float, float]]
) -> tuple[IdentifiabilityFinding, ...]:
    converged = [r for r in start_results if r.converged]
    if len(converged) < 2:
        return ()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    ranges = np.where(hi > lo, hi - lo, 1.0)

    findings = []
    seen_pairs: set[tuple[int, int]] = set()
    for a in range(len(converged)):
        for b in range(a + 1, len(converged)):
            va = np.asarray(converged[a].final_vector)
            vb = np.asarray(converged[b].final_vector)
            distance = float(np.linalg.norm((va - vb) / ranges))
            obj_a, obj_b = converged[a].objective, converged[b].objective
            denom = max(abs(obj_a), abs(obj_b), 1e-12)
            relative_obj_diff = abs(obj_a - obj_b) / denom
            if (
                distance > _DEGENERATE_PARAMETER_DISTANCE_THRESHOLD
                and relative_obj_diff < _DEGENERATE_OBJECTIVE_RELATIVE_THRESHOLD
                and (a, b) not in seen_pairs
            ):
                seen_pairs.add((a, b))
                findings.append(
                    IdentifiabilityFinding(
                        finding_id=f"degenerate-objective:{a}:{b}",
                        category=IdentifiabilityCategory.DEGENERATE_OBJECTIVE,
                        severity=IdentifiabilitySeverity.CRITICAL,
                        summary=(
                            f"Multi-start results {a} and {b} converge to substantially "
                            f"different parameter vectors (normalized distance {distance:.2f}) "
                            f"with nearly identical objective values (relative difference "
                            f"{relative_obj_diff:.2e})."
                        ),
                        explanation=(
                            "Widely different parameter combinations fit the supplied "
                            "observations almost equally well -- the objective surface is "
                            "essentially flat across a large region, so no single fitted value "
                            "is a trustworthy unique answer."
                        ),
                    )
                )
    return tuple(findings)


def insufficient_observations_finding(
    n_train_points: int, n_free_parameters: int
) -> IdentifiabilityFinding | None:
    if n_train_points >= n_free_parameters:
        return None
    return IdentifiabilityFinding(
        finding_id="insufficient-observations",
        category=IdentifiabilityCategory.INSUFFICIENT_OBSERVATIONS,
        severity=IdentifiabilitySeverity.CRITICAL,
        summary=(
            f"Only {n_train_points} training observation point(s) for "
            f"{n_free_parameters} fitted degree(s) of freedom."
        ),
        explanation=(
            "Fitting more free parameters than there are independent training data points "
            "produces an underdetermined system -- any resulting estimate would be an "
            "arbitrary artifact of the optimizer's own starting point, not a meaningful fit."
        ),
    )


__all__ = [
    "bound_hit_findings",
    "correlation_findings",
    "degenerate_objective_findings",
    "insufficient_observations_finding",
    "is_at_bound",
    "low_sensitivity_findings",
]
