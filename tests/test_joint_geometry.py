"""Regressions for the joint convex-mixture slope range.

The original defect stored only per-pair marginal intervals and certified them
as if the six pairs could be extremised independently.  Convex mixtures tie
the pairs together, so those pointwise intervals were too wide.  These tests
pin the joint geometry: the fixed piece must preserve exactly the relative
readings a real convex mixture can realise, and the third-smallest slope range
must be exact on that hull.
"""

from fractions import Fraction

from archive.publication import publish
from issue.transport import decode_envelope
from review.exact_lp import feasible, maximize
from review.geometry import (
    convex_contains,
    marginal_pair_ranges,
    relative_point,
    relative_points,
    slope_vector,
    third_smallest_range,
)
from review.slope_range import piece_relative_points
from review.slopes import third_smallest_slope
from strips.stations import make_station_map
from tests import LabTestCase, sample

F = Fraction
STATIONS = make_station_map(('P4', 'P2', 'P1', 'P3'), (0, 1, 2, 3))


def mixture(points, weights):
    weights = [F(weight) for weight in weights]
    if len(weights) != len(points):
        raise ValueError('one weight per point')
    return tuple(
        sum(weight * point[axis] for weight, point in zip(weights, points))
        for axis in range(3)
    )


def third_of(point):
    return sorted(slope_vector(STATIONS, point))[2]


class JointGeometryTests(LabTestCase):
    def test_relative_points_drop_anchor_and_duplicates(self):
        rows = [(0, 1, 2, 3), (5, 6, 7, 8), (0, 1, 2, 4)]
        self.assertEqual(relative_point(rows[0]), (1, 2, 3))
        self.assertEqual(relative_points(rows), ((1, 2, 3), (1, 2, 4)))

    def test_piece_preserves_all_and_only_mixtures(self):
        data = sample('boundaries')
        _, _, simplex_intake = self.accepted(data['simplex'])
        _, _, cube_intake = self.accepted(data['cube'])
        self.take_source_offline()
        from review.repository import IntakeRepository

        intake = IntakeRepository(self.center)
        simplex_points = piece_relative_points(intake.get(simplex_intake))
        cube_points = piece_relative_points(intake.get(cube_intake))

        # A real convex mixture of two stored scenarios is inside the hull,
        # even though it appears in no scenario row.
        left, right = simplex_points[1], simplex_points[2]
        interior = mixture((left, right), ('1/3', '2/3'))
        self.assertTrue(convex_contains(simplex_points, interior))
        self.assertTrue(convex_contains(cube_points, interior))

        # All and only: the cube witness is reachable by the cube and not by
        # the simplex, matching the accepted evidence.
        witness = (F(1, 3), F(2, 3), F(1))
        self.assertFalse(convex_contains(simplex_points, witness))
        self.assertTrue(convex_contains(cube_points, witness))

        # Anything outside the hull is rejected.
        outside = (F(2), F(0), F(0))
        self.assertFalse(convex_contains(simplex_points, outside))
        self.assertFalse(convex_contains(cube_points, outside))

    def test_frozen_body_keeps_relative_points_only(self):
        _, wire = self.frozen(sample('boundaries')['cube'])
        body = decode_envelope(wire).body
        self.assertIn('relative_points', body)
        self.assertEqual(len(body['relative_points']), 15)  # (0,0,0) plus the cube's 15
        self.assertEqual(len(body['pair_ranges']), 6)
        # No absolute readings or calibration rows cross the boundary.
        self.assertEqual(set(body), {'stations', 'relative_points', 'pair_ranges'})

    def test_exact_bounds_have_witnesses(self):
        # Each endpoint is reached by an explicit real convex mixture, so the
        # interval is attainable, not merely an outer approximation.
        data = sample('boundaries')
        simplex = relative_points(data['simplex'])
        cube = relative_points(data['cube'])
        witnesses = [
            (simplex, ('0', '0', '1/4', '0', '3/4'), F(1, 4)),
            (simplex, ('0', '1', '0', '0', '0'), F(-1, 3)),
            (cube, ('0', '0', '0', '2/3', '0', '1/3', '0', '0',
                    '0', '0', '0', '0', '0', '0', '0'), F(1, 3)),
            (cube, ('0', '0', '0', '0', '0', '0', '0', '0',
                    '1/2', '0', '0', '0', '1/2', '0', '0'), F(-1, 2)),
        ]
        for points, group in ((simplex, witnesses[:2]), (cube, witnesses[2:])):
            expected = sorted((third_of(mixture(points, weights)), value) for _, weights, value in group)
            lower, upper = third_smallest_range(STATIONS, points)
            self.assertEqual([value for _, value in expected], [lower, upper])

    def test_joint_range_is_tighter_than_marginal_intervals(self):
        # The bug signature: treating the six pointwise intervals as independent
        # would report a wider third-smallest range than the joint hull allows.
        data = sample('boundaries')
        points = relative_points(data['simplex'])
        lower, upper = third_smallest_range(STATIONS, points)
        marginal = marginal_pair_ranges(STATIONS, points)
        naive_lower = sorted(bound[0] for bound in marginal)[2]
        naive_upper = sorted(bound[1] for bound in marginal)[2]
        self.assertEqual((lower, upper), (F(-1, 3), F(1, 4)))
        # The joint floor is higher and the joint ceiling lower than the naive
        # pointwise order statistics: the defect reported exactly that wider box.
        self.assertGreater(lower, naive_lower)
        self.assertLess(upper, naive_upper)
        self.assertEqual((naive_lower, naive_upper), (F(-1), F(1, 2)))

    def test_nonlinear_offsets_are_exact_and_attained(self):
        points = relative_points([(0, 0, 0, 0)])
        # Compensation shifts the relative readings, not the slopes: the shift
        # (1, 2, 3) is itself a perfectly linear profile, so every slope is 1.
        lower, upper = third_smallest_range(STATIONS, points, (1, 2, 3))
        self.assertEqual((lower, upper), (F(1), F(1)))
        self.assertEqual(third_smallest_slope(STATIONS, (0, 0, 0, 0), (1, 2, 3, 4)), F(1))
        # A genuinely uneven compensation is not linear; the range stays exact.
        lower, upper = third_smallest_range(STATIONS, points, (0, 0, 0, 3))
        self.assertEqual((lower, upper), (F(0), F(0)))

    def test_ties_keep_their_own_rank(self):
        # All six slopes equal 1: the third smallest is 1, not a deduplicated
        # third distinct value.
        points = relative_points([(0, 1, 2, 3)])
        self.assertEqual(len(set(slope_vector(STATIONS, points[0]))), 1)
        self.assertEqual(third_smallest_range(STATIONS, points), (F(1), F(1)))

    def test_duplicate_scenarios_do_not_change_the_range(self):
        rows = sample('boundaries')['simplex']
        doubled = rows + rows[:2]
        self.assertEqual(relative_points(rows), relative_points(doubled))
        self.assertEqual(
            third_smallest_range(STATIONS, relative_points(rows)),
            third_smallest_range(STATIONS, relative_points(doubled)),
        )

    def test_offsets_on_mixed_calibration(self):
        data = sample('boundaries')
        _, wire, received = self.accepted(data['simplex'])
        offsets = (0, '1/7', '2/7', '3/7')
        self.take_source_offline()
        signed = publish(self.center, received, offsets, 1)
        shift = (F(1, 7), F(2, 7) - 0, F(3, 7))
        expected = third_smallest_range(STATIONS, relative_points(data['simplex']), shift)
        self.assertEqual((signed.report.bounds.lower, signed.report.bounds.upper), expected)
        self.assertEqual(signed.report.bounds.lower, F(-4, 21))
        self.assertEqual(signed.report.bounds.upper, F(11, 28))

    def test_reported_range_is_exactly_attainable(self):
        # Both endpoints are attained by concrete mixtures over the vertex set,
        # so the reported interval is the exact image, not just a bound.
        data = sample('boundaries')
        for rows in (data['simplex'], data['cube']):
            points = relative_points(rows)
            lower, upper = third_smallest_range(STATIONS, points)
            reached = {third_of(point) for point in points}
            reached |= {third_of(mixture(points, weights)) for weights in self._witnesses(rows)}
            self.assertIn(lower, reached)
            self.assertIn(upper, reached)

    @staticmethod
    def _witnesses(rows):
        # Stored vertices together with exact interior witnesses known to attain
        # the endpoints of the fixture hulls.
        points = relative_points(rows)
        for index in range(len(points)):
            yield tuple('1' if j == index else '0' for j in range(len(points)))
        if len(points) == 5:
            yield ('0', '0', '1/4', '0', '3/4')
        if len(points) == 15:
            cube = ('0', '0', '0', '2/3', '0', '1/3', '0', '0',
                    '0', '0', '0', '0', '0', '0', '0')
            yield cube
            yield ('0', '0', '0', '0', '0', '0', '0', '0',
                   '1/2', '0', '0', '0', '1/2', '0', '0')


class ExactLpTests(LabTestCase):
    def brute(self, cost, matrix, rhs):
        from itertools import combinations

        count = len(cost)
        constraints = [[F(value) for value in row] for row in matrix]
        limits = [F(value) for value in rhs]
        constraints += [[F(-1) if j == i else F(0) for j in range(count)] for i in range(count)]
        limits += [F(0)] * count
        best = None
        for chosen in combinations(range(len(constraints)), count):
            square = [constraints[i][:] for i in chosen]
            values = [limits[i] for i in chosen]
            invertible = True
            for column in range(count):
                pivot = next((r for r in range(column, count) if square[r][column] != 0), None)
                if pivot is None:
                    invertible = False
                    break
                square[column], square[pivot] = square[pivot], square[column]
                values[column], values[pivot] = values[pivot], values[column]
                scale = square[column][column]
                square[column] = [value / scale for value in square[column]]
                values[column] /= scale
                for r in range(count):
                    if r != column and square[r][column] != 0:
                        factor = square[r][column]
                        square[r] = [x - factor * y for x, y in zip(square[r], square[column])]
                        values[r] -= factor * values[column]
            if not invertible:
                continue
            if any(value < 0 for value in values):
                continue
            if any(sum(constraints[i][j] * values[j] for j in range(count)) > limits[i]
                   for i in range(len(constraints))):
                continue
            objective = sum(cost[j] * values[j] for j in range(count))
            best = objective if best is None or objective > best else best
        return best

    def test_maximize_against_vertex_enumeration(self):
        cases = [
            ([1, 1], [[1, 2], [3, 1]], [4, 6]),
            ([0, 0, 1], [[1, 1, 1], [1, -1, 0]], [1, 0]),
            ([2, 3], [[1, 1], [1, -1]], [1, 0]),
        ]
        for cost, matrix, rhs in cases:
            with self.subTest(cost=cost):
                expected = self.brute(list(map(F, cost)), matrix, list(map(F, rhs)))
                result = maximize(list(map(F, cost)), matrix, list(map(F, rhs)))
                self.assertIsNotNone(result)
                self.assertEqual(result[0], expected)

    def test_negative_rhs_feasibility(self):
        self.assertFalse(feasible([[F(1)]], [F(-1)]))
        self.assertTrue(feasible([[F(-1)]], [F(-1)]))
        self.assertTrue(feasible([[F(1), F(-1)]], [F(0)]))
        self.assertFalse(feasible([[F(1), F(1)]], [F(-1)]))

    def test_unbounded_reports_none(self):
        self.assertIsNone(maximize([F(1)], [], []))
