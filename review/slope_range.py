from dataclasses import dataclass
from fractions import Fraction

from calibration.scenarios import rational_row
from review.repository import AcceptedCalibration
from review.request import ReviewRequest
from review.slopes import pair_slopes


@dataclass(frozen=True, slots=True)
class SlopeRange:
    lower: Fraction
    upper: Fraction


def marginal_slope_ranges(calibration: AcceptedCalibration) -> tuple[SlopeRange, ...]:
    rows = calibration.envelope.body.get("pair_ranges")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("piece has no usable slope ranges")
    ranges = tuple(SlopeRange(*rational_row(row, 2)) for row in rows)
    if any(bound.lower > bound.upper for bound in ranges):
        raise ValueError("slope range is reversed")
    return ranges


def supports_relative_readings(calibration: AcceptedCalibration, readings) -> bool:
    target = rational_row(readings, 3)
    slopes = pair_slopes(calibration.envelope.station_map, (0, *target))
    return all(
        bound.lower <= slope <= bound.upper
        for bound, slope in zip(marginal_slope_ranges(calibration), slopes, strict=True)
    )


def certify_slope_range(calibration: AcceptedCalibration, request: ReviewRequest) -> SlopeRange:
    if calibration.intake_ref != request.intake_ref:
        raise ValueError("request does not name this accepted piece")
    ranges = marginal_slope_ranges(calibration)
    offsets = pair_slopes(calibration.envelope.station_map, request.offsets)
    compensated = tuple(
        SlopeRange(bound.lower + offset, bound.upper + offset)
        for bound, offset in zip(ranges, offsets, strict=True)
    )
    lower = sorted(bound.lower for bound in compensated)[2]
    upper = sorted(bound.upper for bound in compensated)[2]
    return SlopeRange(lower, upper)
