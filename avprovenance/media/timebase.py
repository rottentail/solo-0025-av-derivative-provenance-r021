"""Exact rational media time and deterministic boundary resolution.

An exact time is an object with integer ``numerator`` and ``denominator``
fields and denotes ``numerator / denominator`` seconds. It is evaluated with
exact rational arithmetic only, so no float conversion or rounding can move a
boundary by a sample or a frame.

A boundary time identifies the start instant of the sample or frame that
contains it. The boundary index is::

    floor(seconds * rate)

where the rate is the sample rate for audio and ``rate_numerator /
rate_denominator`` for video. Because every boundary is resolved independently
from the same source origin, two adjacent selections that share an exact
boundary resolve to the same index and therefore neither duplicate nor drop a
sample/frame.

Normalized exact trim operations record the interpretation they were created
with. Reproduction refuses an exact trim that does not carry that record, so a
stored plan can never silently change to another rounding policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any


EXACT_TIME_MODE = "exact"
EXACT_BOUNDARY_RULE = "floor"


class MediaTimeError(ValueError):
    """A media-time coordinate or its recorded interpretation is unusable."""


def seconds(value: Any) -> Fraction:
    """Return an exact rational number of seconds from a media-time object."""
    if not isinstance(value, Mapping):
        raise MediaTimeError("A media-time value must be an object.")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise MediaTimeError("A media-time numerator must be an integer.")
    if isinstance(denominator, bool) or not isinstance(denominator, int):
        raise MediaTimeError("A media-time denominator must be an integer.")
    if numerator < 0:
        raise MediaTimeError("A media-time numerator must not be negative.")
    if denominator <= 0:
        raise MediaTimeError("A media-time denominator must be positive.")
    return Fraction(numerator, denominator)


def floor_index(
    value: Fraction,
    rate_numerator: int,
    rate_denominator: int = 1,
) -> int:
    """Resolve an exact time to the index of the sample/frame containing it."""
    if rate_numerator <= 0 or rate_denominator <= 0:
        raise MediaTimeError("A media rate must be positive.")
    scaled = value * Fraction(rate_numerator, rate_denominator)
    return scaled.numerator // scaled.denominator


def exact_trim_bounds(
    operation: Mapping[str, Any],
    *,
    rate_numerator: int,
    rate_denominator: int = 1,
) -> tuple[int, int]:
    """Return the half-open ``[start, end)`` sample/frame bounds of a trim.

    The end boundary is derived from the exact rational sum of the start time
    and the duration, or from the exact end time. It is never derived from
    already-rounded start/duration indices, which keeps adjacent selections
    seamless.
    """
    if operation.get("time_mode") != EXACT_TIME_MODE:
        raise MediaTimeError(
            "The stored media_trim does not record exact time coordinates."
        )
    if operation.get("boundary_rule") != EXACT_BOUNDARY_RULE:
        raise MediaTimeError(
            "The stored media_trim does not record a supported boundary rule."
        )
    start_value = seconds(operation.get("start_time"))
    if "end_time" in operation:
        if "duration_time" in operation:
            raise MediaTimeError(
                "The stored media_trim records both an end time and a duration."
            )
        end_value = seconds(operation.get("end_time"))
    elif "duration_time" in operation:
        end_value = start_value + seconds(operation.get("duration_time"))
    else:
        raise MediaTimeError("The stored media_trim has no end or duration time.")
    if end_value <= start_value:
        raise MediaTimeError("The stored media_trim has an empty time range.")
    return (
        floor_index(start_value, rate_numerator, rate_denominator),
        floor_index(end_value, rate_numerator, rate_denominator),
    )
