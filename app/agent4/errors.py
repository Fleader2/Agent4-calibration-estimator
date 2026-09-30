"""Agent 4's own exception hierarchy.

Mirrors the sibling repositories' own per-package error convention (``app.agent3.errors`` on
the sibling ``agent3-simulation-diagnostics`` repository): a narrow, typed hierarchy raised only
for genuine structural/contract problems Agent 4 itself detects before ever attempting
calibration -- never used to represent a *calibration outcome* (a failed optimization, an
unidentifiable model, insufficient data are all reported as data --
``CalibrationStatus``/``Agent4Report`` -- never raised as Python exceptions, since they are
expected, first-class outcomes this package exists to describe, not programming errors).
"""

from __future__ import annotations


class Agent4Error(Exception):
    """Base class for every exception this package raises."""


class HandoffContractError(Agent4Error):
    """The supplied Agent 2/Agent 3 -> Agent 4 handoff dict is missing a required field, has a
    field of the wrong type, or otherwise fails to satisfy ``app.agent4.handoff``'s own
    contract. Never raised for a handoff that is merely *unhelpful* (e.g. a model with zero
    calibratable parameters) -- those are legitimate, if limited, inputs Agent 4 loads and then
    reports on. Raised only when the data cannot be parsed into the contract's own required
    shape at all.
    """


class CalibrationRequestError(Agent4Error):
    """Raised only when a ``CalibrationRequest`` cannot even be validated against the handoff
    it targets (e.g. a target references a species/parameter id that does not exist in the
    handoff at all). An eligibility *violation* (a protected-provenance parameter requested
    without authorization, a fixed parameter requested, missing bounds) is never raised as this
    exception -- it is reported as ``CalibrationStatus.INVALID_REQUEST`` in the resulting
    ``Agent4Report``, since a caller submitting an unauthorized request is exactly the kind of
    first-class, expected outcome this package exists to refuse safely and explain, not an
    internal programming error.
    """


__all__ = ["Agent4Error", "CalibrationRequestError", "HandoffContractError"]
