from calibration.revisions import ScenarioRevision
from review.geometry import marginal_pair_ranges, relative_points
from strips.stations import StationMap


def freeze_relative_evidence(revision: ScenarioRevision, stations: StationMap) -> dict:
    """Preserve exactly the relative readings any real convex mixture can realise.

    The scenario points span the mixture hull, so keeping every distinct
    relative point keeps the whole feasible set and nothing more.  The
    absolute readings and the calibration record stay in the source database;
    only the relative geometry crosses into the fixed piece.
    """
    points = relative_points(revision.batch.rows)
    return {
        "stations": stations.to_payload(),
        "relative_points": [[str(value) for value in point] for point in points],
        "pair_ranges": [
            [str(lower), str(upper)]
            for lower, upper in marginal_pair_ranges(stations, points)
        ],
    }
