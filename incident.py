import json
import sqlite3
from contextlib import closing
from pathlib import Path

from archive.history import read
from archive.publication import publish
from archive.repository import ArchiveRepository
from calibration.corrections import record
from calibration.repository import RevisionRepository
from issue.freeze import issue
from issue.repository import FrozenPieceRepository
from issue.transport import decode_envelope
from review.intake import receive
from review.repository import IntakeRepository
from review.slope_range import supports_relative_readings
from strips.catalog import register
from strips.repository import StripRepository


def main() -> int:
    sample = Path(__file__).resolve().with_name("samples") / "boundaries.json"
    data = json.loads(sample.read_text(encoding="utf-8"))
    offsets = (0, 0, 0, 0)
    with closing(sqlite3.connect(":memory:")) as center:
        IntakeRepository(center).initialize()
        ArchiveRepository(center).initialize()
        with closing(sqlite3.connect(":memory:")) as source:
            StripRepository(source).initialize()
            RevisionRepository(source).initialize()
            FrozenPieceRepository(source).initialize()
            strip = register(source, "HC-incident", ("P4", "P2", "P1", "P3"))
            initial = record(source, strip, data["simplex"])
            correction = record(source, strip, data["cube"], previous=initial)
            wires = [issue(source, revision) for revision in (initial, correction)]
            intakes = [receive(center, wire) for wire in wires]

        actual = []
        for wire, intake in zip(wires, intakes, strict=True):
            piece = decode_envelope(wire)
            accepted = IntakeRepository(center).get(intake)
            supported = supports_relative_readings(accepted, data["cube_only_target"])
            signed = publish(center, intake, offsets, data["limit"])
            retained = read(center, signed.report.report_id)
            report = retained.report
            actual.append({
                "target_supported": supported,
                "range": [str(report.bounds.lower), str(report.bounds.upper)],
                "exempt": report.exempt,
                "original_publication": retained == signed,
                "right_parent": (report.intake_ref, report.piece_id, report.revision_id)
                == (intake, piece.piece_id, piece.revision_id),
            })
    expected = [
        {"target_supported": False, "range": ["-1/3", "1/4"], "exempt": True,
         "original_publication": True, "right_parent": True},
        {"target_supported": True, "range": ["-1/2", "1/3"], "exempt": False,
         "original_publication": True, "right_parent": True},
    ]
    print(json.dumps({"limit": data["limit"], "expected": expected, "actual": actual}, sort_keys=True))
    if actual != expected:
        print("STRIP_TREND_RANGE_MISISSUE")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
