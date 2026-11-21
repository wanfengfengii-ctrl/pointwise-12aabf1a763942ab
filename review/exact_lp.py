"""Exact rational linear programming for the centre's offline review.

The review programs are tiny (at most sixteen points, seven constraints) and
must stay exact, so this is a Bland's-rule primal simplex over ``Fraction``
with a standard two-phase start.  It is deliberately not a general solver:
it answers ``max c.x`` subject to ``A x <= b`` and ``x >= 0`` and reports
infeasibility or unboundedness as ``None``.
"""

from fractions import Fraction

F = Fraction


def _pivot(rows, basis, objective, pivot_row, pivot_col):
    scale = rows[pivot_row][pivot_col]
    rows[pivot_row] = [entry / scale for entry in rows[pivot_row]]
    for index, row in enumerate(rows):
        if index != pivot_row and row[pivot_col] != 0:
            factor = row[pivot_col]
            rows[index] = [left - factor * right for left, right in zip(row, rows[pivot_row])]
    factor = objective[pivot_col]
    if factor != 0:
        pivot = rows[pivot_row]
        for index in range(len(objective)):
            objective[index] -= factor * pivot[index]
    basis[pivot_row] = pivot_col


def _optimize(rows, basis, objective, allowed):
    columns = len(objective) - 1
    while True:
        entering = next((j for j in range(columns) if j in allowed and objective[j] < 0), None)
        if entering is None:
            return True
        leaving = None
        best = None
        for index, row in enumerate(rows):
            coefficient = row[entering]
            if coefficient > 0:
                ratio = row[-1] / coefficient
                if (best is None or ratio < best
                        or (ratio == best and basis[index] < basis[leaving])):
                    best, leaving = ratio, index
        if leaving is None:
            return False
        _pivot(rows, basis, objective, leaving, entering)


def _objective(costs, basis, rows, width):
    objective = [-costs[j] if j < len(costs) else F(0) for j in range(width)] + [F(0)]
    for index, basic in enumerate(basis):
        coefficient = costs[basic] if basic < len(costs) else F(0)
        if coefficient != 0:
            for j in range(width + 1):
                objective[j] += coefficient * rows[index][j]
    return objective


def maximize(cost, matrix, rhs):
    """Return ``(optimum, argmax)`` for ``max cost*x`` with ``A x <= b``, ``x >= 0``.

    ``None`` means the program is infeasible or unbounded.
    """
    count, constraints = len(cost), len(matrix)
    cost = [F(value) for value in cost]
    rhs = [F(value) for value in rhs]
    matrix = [[F(value) for value in row] for row in matrix]

    negatives = [index for index in range(constraints) if rhs[index] < 0]
    artificials = len(negatives)
    structural = count + constraints
    width = structural + artificials
    rows, basis = [], []
    for index in range(constraints):
        sign = -1 if index in negatives else 1
        row = [F(0)] * (width + 1)
        for j in range(count):
            row[j] = sign * matrix[index][j]
        row[count + index] = F(sign)
        row[-1] = sign * rhs[index]
        rows.append(row)
        basis.append(count + index)
    for offset, index in enumerate(negatives):
        column = structural + offset
        rows[index][column] = F(1)
        basis[index] = column

    if artificials:
        phase_one = [F(0)] * count + [F(0)] * constraints + [F(-1)] * artificials
        objective = _objective(phase_one, basis, rows, width)
        if not _optimize(rows, basis, objective, set(range(structural))):
            return None
        # A zero optimum for -sum(artificials) means the original program is feasible.
        if objective[-1] != 0:
            return None
        for index in range(constraints):
            if basis[index] >= structural:
                swap = next(
                    (j for j in range(structural) if j not in basis and rows[index][j] != 0),
                    None,
                )
                if swap is not None:
                    _pivot(rows, basis, objective, index, swap)
        keep = [index for index in range(constraints) if basis[index] < structural]
        rows = [rows[index][:structural] + [rows[index][-1]] for index in keep]
        basis = [basis[index] for index in keep]

    costs = [F(value) for value in cost] + [F(0)] * constraints
    objective = _objective(costs, basis, rows, structural)
    if not _optimize(rows, basis, objective, set(range(structural))):
        return None
    solution = [F(0)] * count
    for index, basic in enumerate(basis):
        if basic < count:
            solution[basic] = rows[index][-1]
    return objective[-1], solution


def feasible(matrix, rhs):
    """Return whether ``A x <= b`` with ``x >= 0`` has any solution."""
    if not matrix:
        return True
    return maximize([F(0)] * len(matrix[0]), matrix, rhs) is not None
