"""Deterministic multi-start optimization (``optimization-v1``).

Uses ``scipy.optimize.minimize`` (bounded, gradient-based; ``L-BFGS-B`` by default) from a
fixed, deterministic set of starting points -- never a random draw, so no seed is needed for
reproducibility and the same ``CalibrationRequest`` always produces the same
``CalibrationResult``. No Bayesian/MCMC machinery is used in Version 1, per the task's own
explicit instruction to avoid heavy dependencies unless strictly necessary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from app.agent4.handoff import Agent2Model
from app.agent4.objective import evaluate_vector
from app.agent4.types import CalibrationTarget, Observation, OptimizerConfig


@dataclass(frozen=True, slots=True)
class StartResult:
    """The outcome of optimizing from one deterministic starting point."""

    start_vector: tuple[float, ...]
    final_vector: tuple[float, ...]
    objective: float
    converged: bool
    message: str


def generate_start_vectors(
    baseline_vector: np.ndarray,
    bounds: list[tuple[float, float]],
    n_multistarts: int,
) -> list[np.ndarray]:
    """``n_multistarts`` deterministic starting points: start 0 is ``baseline_vector`` (clipped
    into bounds); every subsequent start is a fixed, staggered interpolation fraction across
    each dimension's own bounds -- staggered per dimension (via a fixed cyclic offset) so
    different starts probe different corners of the search space without ever calling a random
    number generator.
    """
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    starts = [np.clip(baseline_vector, lo, hi)]

    remaining = max(n_multistarts - 1, 0)
    for i in range(remaining):
        frac_base = (i + 1) / (remaining + 1)
        fractions = np.array(
            [min(max((frac_base + d / (dim + 1)) % 1.0, 0.05), 0.95) for d in range(dim)]
        )
        starts.append(lo + fractions * (hi - lo))
    return starts[:n_multistarts] if n_multistarts >= 1 else starts


def run_multistart_optimization(
    agent2_model: Agent2Model,
    targets: tuple[CalibrationTarget, ...],
    bounds: list[tuple[float, float]],
    baseline_vector: np.ndarray,
    train_observations: tuple[Observation, ...],
    species_id_map: dict[str, str],
    config: OptimizerConfig,
) -> list[StartResult]:
    """Run one bounded optimization from each of ``config.n_multistarts`` deterministic
    starting points and return every start's own result, in start order -- the caller
    (``app.agent4.pipeline``) selects the best converged start."""

    def objective(vector: np.ndarray) -> float:
        return evaluate_vector(
            agent2_model, targets, vector, train_observations, species_id_map
        ).objective

    starts = generate_start_vectors(baseline_vector, bounds, config.n_multistarts)
    results: list[StartResult] = []
    for start in starts:
        outcome = minimize(
            objective,
            start,
            method=config.method,
            bounds=bounds,
            options={"maxiter": config.max_iterations, "ftol": config.function_tolerance},
        )
        results.append(
            StartResult(
                start_vector=tuple(float(v) for v in start),
                final_vector=tuple(float(v) for v in outcome.x),
                objective=float(outcome.fun),
                converged=bool(outcome.success),
                message=str(outcome.message),
            )
        )
    return results


__all__ = ["StartResult", "generate_start_vectors", "run_multistart_optimization"]
