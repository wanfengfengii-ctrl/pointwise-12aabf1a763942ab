from dataclasses import dataclass
from fractions import Fraction

from calibration.scenarios import rational_row
from review.geometry import convex_contains, third_smallest_range
from review.repository import AcceptedCalibration
from review.request import ReviewRequest

F = Fraction


@dataclass(frozen=True, slots=True)
class SlopeRange:
    lower: Fraction
    upper: Fraction


def piece_relative_points(calibration: AcceptedCalibration) -> tuple[tuple[Fraction, ...], ...]:
    """The distinct relative points preserved in the accepted piece."""
    body = calibration.envelope.body
    raw = body.get("relative_points")
    if not isinstance(raw, list) or not raw:
        raise ValueError("piece has no usable relative evidence")
    points = tuple(rational_row(point, 3) for point in raw)
    if len(set(points)) != len(points):
        raise ValueError("piece repeats a relative point")
    return points


def marginal_slope_ranges(calibration: AcceptedCalibration) -> tuple[SlopeRange, ...]:
    body = calibration.envelope.body
    rows = body.get("pair_ranges")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("piece has no usable slope ranges")
    ranges = tuple(SlopeRange(*rational_row(row, 2)) for row in rows)
    if any(bound.lower > bound.upper for bound in ranges):
        raise ValueError("slope range is reversed")
    return ranges


def relative_shift(offsets) -> tuple[Fraction, Fraction, Fraction]:
    """Compensation acts on the readings, so it shifts their relative point."""
    values = tuple(F(value) for value in offsets)
    return (values[1] - values[0], values[2] - values[0], values[3] - values[0])


def supports_relative_readings(calibration: AcceptedCalibration, readings) -> bool:
    """Whether the target relative readings lie in the piece's mixture hull."""
    target = rational_row(readings, 3)
    return convex_contains(piece_relative_points(calibration), target)


def certify_slope_range(calibration: AcceptedCalibration, request: ReviewRequest) -> SlopeRange:
    if calibration.intake_ref != request.intake_ref:
        raise ValueError("request does not name this accepted piece")
    points = piece_relative_points(calibration)
    shift = relative_shift(request.offsets)
    lower, upper = third_smallest_range(
        calibration.envelope.station_map, points, shift
    )
    return SlopeRange(lower, upper)
