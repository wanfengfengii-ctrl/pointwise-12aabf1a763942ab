"""Exact relative-reading geometry for frozen pieces.

A revision is a finite set of four-point scenarios.  A real convex mixture
picks nonnegative weights summing to one, so the relative readings a mixture
can realise are exactly the convex hull of the scenarios' relative readings.
Projecting ``(r0, r1, r2, r3)`` to ``(r1 - r0, r2 - r0, r3 - r0)`` is linear,
which is why the relative evidence can be kept in the piece while the absolute
readings stay in the source.

Every value here is an exact :class:`~fractions.Fraction`; no floats enter.
"""

from fractions import Fraction
from itertools import combinations

from review.exact_lp import feasible, maximize
from strips.stations import StationMap

F = Fraction

# The six unordered station pairs, in ``combinations(range(4), 2)`` order.
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def relative_point(readings) -> tuple[Fraction, Fraction, Fraction]:
    """Project four readings onto the three differences from station 0."""
    values = [F(reading) for reading in readings]
    if len(values) != 4:
        raise ValueError('exactly four readings are required')
    return (values[1] - values[0], values[2] - values[0], values[3] - values[0])


def relative_points(rows) -> tuple[tuple[Fraction, ...], ...]:
    """Return the distinct relative points of a batch of scenarios, in order."""
    points: list[tuple[Fraction, ...]] = []
    for row in rows:
        point = relative_point(row)
        if point not in points:
            points.append(point)
    return tuple(points)


def convex_contains(points, target) -> bool:
    """Whether ``target`` is a real convex mixture of ``points``.

    Membership is decided exactly: ``target = sum(weight * point)`` with
    nonnegative weights summing to one.  The last point is used as the anchor,
    so the unknowns are the weights of the remaining points.
    """
    target = tuple(F(value) for value in target)
    if len(target) != 3:
        raise ValueError('a relative target must have three coordinates')
    points = tuple(tuple(F(value) for value in point) for point in points)
    if not points:
        raise ValueError('no relative points to mix')
    if len(points) == 1:
        return target == points[0]
    anchor = points[-1]
    count = len(points) - 1
    matrix = [[F(1)] * count]
    rhs = [F(1)]
    for axis in range(3):
        difference = [points[j][axis] - anchor[axis] for j in range(count)]
        offset = target[axis] - anchor[axis]
        matrix.append(difference)
        rhs.append(offset)
        matrix.append([-value for value in difference])
        rhs.append(-offset)
    return feasible(matrix, rhs)


def _slope(distances, point, left: int, right: int) -> Fraction:
    if left == 0:
        difference = point[right - 1]
    else:
        difference = point[right - 1] - point[left - 1]
    return difference / distances[(left, right)]


def _distances(stations: StationMap) -> dict[tuple[int, int], int]:
    return {(left, right): stations.distance(left, right) for left, right in PAIRS}


def slope_vector(stations: StationMap, point) -> tuple[Fraction, ...]:
    distances = _distances(stations)
    return tuple(_slope(distances, point, left, right) for left, right in PAIRS)


def marginal_pair_ranges(stations: StationMap, points) -> tuple[tuple[Fraction, Fraction], ...]:
    """Per-pair extremes over the stored scenario points (pointwise, not joint)."""
    distances = _distances(stations)
    points = tuple(tuple(F(value) for value in point) for point in points)
    ranges = []
    for left, right in PAIRS:
        values = [_slope(distances, point, left, right) for point in points]
        ranges.append((min(values), max(values)))
    return tuple(ranges)


def third_smallest_range(stations: StationMap, points, shift=(0, 0, 0)):
    """Exact min and max of the third-smallest pair slope over the hull.

    The per-point compensation is applied at the readings level, so it acts on
    the relative point as the relative shift ``(o1-o0, o2-o0, o3-o0)``.  Ties
    keep their own rank.  The extremes follow from the order-statistic identity
    ``third = max_{|S|=4} min_{k in S} s_k = min_{|T|=3} max_{k in T} s_k``,
    each inner extremum being one tiny exact LP over the convex hull.
    """
    shift = tuple(F(value) for value in shift)
    shifted = [
        tuple(F(point[axis]) + shift[axis] for axis in range(3)) for point in points
    ]
    if not shifted:
        raise ValueError('no relative points to certify')
    vectors = [slope_vector(stations, point) for point in shifted]
    if len(vectors) == 1:
        value = sorted(vectors[0])[2]
        return value, value

    count = len(vectors) - 1
    anchor = vectors[-1]
    delta = [[vectors[j][k] - anchor[k] for j in range(count)] for k in range(6)]
    floor = min(value for vector in vectors for value in vector)
    ceiling = max(value for vector in vectors for value in vector)

    def upper_for(subset):
        matrix = [[F(1)] * count + [F(0)]]
        rhs = [F(1)]
        for k in subset:
            matrix.append([-delta[k][j] for j in range(count)] + [F(1)])
            rhs.append(anchor[k] - floor)
        optimum = maximize([F(0)] * count + [F(1)], matrix, rhs)
        if optimum is None:
            raise ValueError('slope program is not solvable')
        return optimum[0]

    def lower_for(subset):
        matrix = [[F(1)] * count + [F(0)]]
        rhs = [F(1)]
        for k in subset:
            matrix.append([delta[k][j] for j in range(count)] + [F(1)])
            rhs.append(ceiling - anchor[k])
        optimum = maximize([F(0)] * count + [F(1)], matrix, rhs)
        if optimum is None:
            raise ValueError('slope program is not solvable')
        return optimum[0]

    upper = floor + max(upper_for(subset) for subset in combinations(range(6), 4))
    lower = ceiling - max(lower_for(subset) for subset in combinations(range(6), 3))
    return lower, upper
