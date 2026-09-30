"""A deterministic helper for assigning observations to train/validation partitions.

Never a random split: observations are sorted by their own ``observation_id`` and the last
``round(n * validation_fraction)`` of them (by that sort order) are reassigned to
``ObservationPartition.VALIDATION``; every other caller-supplied field is left unchanged. A
caller that wants a specific split (e.g. dedicated validation species/time windows) should
simply construct ``Observation`` objects with their own explicit ``partition`` instead of using
this helper at all -- this function exists only for the common case of "give me *some*
deterministic held-out fraction," never as the only way to express a train/validation split.
"""

from __future__ import annotations

from dataclasses import replace

from app.agent4.types import Observation, ObservationPartition


def split_deterministically(
    observations: tuple[Observation, ...], validation_fraction: float
) -> tuple[Observation, ...]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in [0, 1), got {validation_fraction!r}")
    ordered = sorted(observations, key=lambda o: o.observation_id)
    n_validation = round(len(ordered) * validation_fraction)
    validation_ids = (
        {o.observation_id for o in ordered[len(ordered) - n_validation :]}
        if n_validation
        else set()
    )
    return tuple(
        replace(o, partition=ObservationPartition.VALIDATION)
        if o.observation_id in validation_ids
        else replace(o, partition=ObservationPartition.TRAIN)
        for o in observations
    )


__all__ = ["split_deterministically"]
