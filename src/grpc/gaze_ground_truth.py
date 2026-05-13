from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ComparisonResult:
    compared_pairs: int
    mismatched_pairs: int
    is_mismatch: bool


InteractionPair = Tuple[int, int]


@dataclass(frozen=True)
class LabelInterval:
    start_s: float
    end_s: float
    source_person_id: int
    target_person_id: Optional[int]  # None means "GazeRobot"


class GroundTruthGazeComparator:
    """
    Ground-truth file schema (CSV/TSV):
      - start_time
      - end_time
      - label_type

    Example label_type values:
      - P1GazeP2
      - P3GazeRobot  (treated as "not looking at anyone")
    """

    REQUIRED_COLUMNS = {"start_time", "end_time", "label_type"}
    LABEL_PATTERN = re.compile(r"^P(?P<src>\d+)Gaze(?:(?:P(?P<dst>\d+))|Robot)$", re.IGNORECASE)

    def __init__(self, csv_path: str | Path, person_id_map: Optional[Dict[int, int]] = None) -> None:
        self.csv_path = Path(csv_path)
        self._person_id_map = dict(person_id_map or {})
        self._intervals: List[LabelInterval] = []
        self._people_ids: List[int] = []
        self._load()

    def _map_person_id(self, label_person_id: int) -> int:
        return int(self._person_id_map.get(label_person_id, label_person_id))

    @classmethod
    def _parse_label(cls, label_text: str) -> Tuple[int, Optional[int]]:
        match = cls.LABEL_PATTERN.match(str(label_text).strip())
        if not match:
            raise ValueError(f"Invalid label_type '{label_text}'. Expected forms like P1GazeP2 or P1GazeRobot")
        source_person_id = int(match.group("src"))
        dst_group = match.group("dst")
        target_person_id = int(dst_group) if dst_group is not None else None
        return source_person_id, target_person_id

    def _load(self) -> None:
        with self.csv_path.open("r", encoding="utf-8", newline="") as fh:
            sample = fh.read(2048)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(fh, dialect=dialect)
            if reader.fieldnames is None:
                raise ValueError(f"Ground truth CSV has no header: {self.csv_path}")

            missing = self.REQUIRED_COLUMNS.difference(reader.fieldnames)
            if missing:
                missing_cols = ", ".join(sorted(missing))
                raise ValueError(f"Ground truth CSV missing columns: {missing_cols}")

            people = set()
            intervals: List[LabelInterval] = []
            for row in reader:
                start_s = float(row["start_time"])
                end_s = float(row["end_time"])
                if end_s < start_s:
                    raise ValueError(f"Invalid interval with end_time < start_time: {row}")
                source_person_id, target_person_id = self._parse_label(row["label_type"])
                source_person_id = self._map_person_id(source_person_id)
                if target_person_id is not None:
                    target_person_id = self._map_person_id(target_person_id)
                people.add(source_person_id)
                if target_person_id is not None:
                    people.add(target_person_id)
                intervals.append(
                    LabelInterval(
                        start_s=start_s,
                        end_s=end_s,
                        source_person_id=source_person_id,
                        target_person_id=target_person_id,
                    )
                )

            self._people_ids = sorted(people)
            self._intervals = intervals

    def _expected_pairs_for_timestamp(self, timestamp_s: float) -> Dict[InteractionPair, bool]:
        expected: Dict[InteractionPair, bool] = {}
        # Use all active intervals at this timestamp.
        for interval in self._intervals:
            if interval.start_s <= timestamp_s < interval.end_s:
                src = interval.source_person_id
                if interval.target_person_id is None:
                    # GazeRobot means source is not looking at any person.
                    for dst in self._people_ids:
                        if dst != src:
                            expected[(src, dst)] = False
                else:
                    for dst in self._people_ids:
                        if dst != src:
                            expected[(src, dst)] = (dst == interval.target_person_id)
        return expected

    def compare_update(self, update) -> ComparisonResult:
        timestamp_s = float(getattr(update, "playback_timestamp_ns", 0)) / 1e9
        expected = self._expected_pairs_for_timestamp(timestamp_s)
        if not expected:
            return ComparisonResult(compared_pairs=0, mismatched_pairs=0, is_mismatch=False)

        actual_by_pair: Dict[InteractionPair, bool] = {
            (int(edge.from_person_id), int(edge.to_person_id)): bool(getattr(edge, "is_looking", True))
            for edge in update.interactions
        }
        mismatched = 0
        compared = 0
        for pair, expected_value in expected.items():
            compared += 1
            actual = actual_by_pair.get(pair, False)
            if actual != expected_value:
                mismatched += 1

        return ComparisonResult(
            compared_pairs=compared,
            mismatched_pairs=mismatched,
            is_mismatch=mismatched > 0,
        )

    def available_frames(self) -> Iterable[int]:
        return []


class MismatchDurationGuard:
    def __init__(self, max_mismatch_seconds: float = 2.0) -> None:
        self.max_mismatch_ns = int(max(0.1, max_mismatch_seconds) * 1e9)
        self._mismatch_start_ns: Optional[int] = None

    def update(self, source_timestamp_ns: int, is_mismatch: bool) -> bool:
        """
        Returns True when mismatch duration exceeds configured threshold.
        """
        if not is_mismatch:
            self._mismatch_start_ns = None
            return False

        if self._mismatch_start_ns is None:
            self._mismatch_start_ns = int(source_timestamp_ns)
            return False

        return int(source_timestamp_ns) - self._mismatch_start_ns > self.max_mismatch_ns
