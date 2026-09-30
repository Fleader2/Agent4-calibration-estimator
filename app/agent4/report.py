"""Determining Agent 5 sufficiency for a completed ``CalibrationResult``.

Mirrors ``app.agent3.report.determine_agent4_sufficiency``'s own precedent: this is a
completeness/traceability judgment ("does this report carry enough structured information for
the next agent to act on"), never a scientific-quality judgment about whether the fit is good.
"""

from __future__ import annotations

from app.agent4.types import CalibrationStatus


def determine_agent5_sufficiency(status: CalibrationStatus) -> tuple[bool, str]:
    if status is CalibrationStatus.INVALID_REQUEST:
        return (
            False,
            "The request itself was rejected by the eligibility policy before any simulation "
            "was attempted -- there is no calibration content for Agent 5 to act on beyond the "
            "rejection reason(s) themselves.",
        )
    if status is CalibrationStatus.SIMULATION_FAILED:
        return (
            False,
            "The baseline model (at its original, unperturbed values) could not even be "
            "simulated once -- there is no calibration content for Agent 5 beyond the "
            "simulation failure message itself.",
        )
    if status is CalibrationStatus.INSUFFICIENT_DATA:
        return (
            True,
            "No fit was attempted because there are fewer training observations than fitted "
            "degrees of freedom -- but this is fully disclosed with the exact counts involved, "
            "which is itself useful, actionable information for Agent 5 (or a human) deciding "
            "whether to supply more data or fit fewer targets.",
        )
    if status is CalibrationStatus.OPTIMIZATION_FAILED:
        return (
            True,
            "Every multi-start optimization attempt failed to converge, but every attempt's own "
            "starting point, final point, and solver message is disclosed -- sufficient for "
            "Agent 5 to decide whether to retry with different bounds/starts or supply better "
            "observations.",
        )
    if status is CalibrationStatus.UNIDENTIFIABLE:
        return (
            True,
            "Optimization nominally converged, but the practical-identifiability diagnostics "
            "found the objective to be essentially flat across widely different parameter "
            "combinations -- the estimate itself is not a trustworthy unique answer, but this "
            "is disclosed explicitly rather than silently reported as a confident fit.",
        )
    return (
        True,
        "Optimization converged, every parameter estimate carries its own original value/"
        "provenance alongside the fitted value, residuals are reported for both partitions, and "
        "every identifiability finding (if any) is disclosed -- sufficient structured input for "
        "Agent 5's own, separate process.",
    )


__all__ = ["determine_agent5_sufficiency"]
