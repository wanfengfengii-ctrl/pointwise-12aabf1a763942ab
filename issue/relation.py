from calibration.revisions import ScenarioRevision
from review.slopes import pair_slopes
from strips.stations import StationMap


def freeze_relative_evidence(revision: ScenarioRevision, stations: StationMap) -> dict:
    columns = zip(*(pair_slopes(stations, row) for row in revision.batch.rows))
    return {
        "stations": stations.to_payload(),
        "pair_ranges": [[str(min(column)), str(max(column))] for column in columns],
    }
