from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, TextIO, Tuple


InteractionPair = Tuple[int, int]
MismatchRecord = Tuple[Optional[InteractionPair], Optional[InteractionPair]]


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
    """

    #src and dst are named capturing groups (?P)
    #?: groups multiple regex ops together
    LABEL_PATTERN = re.compile(r"^P(?P<src>\d+)Gaze(?:(?:P(?P<dst>\d+))|Robot)$", re.IGNORECASE)

    def __init__(self, csv_path: str | Path, person_id_map: Optional[Dict[int, int]] = None) -> None:
        self.csv_path = Path(csv_path)
        self._person_id_map = dict(person_id_map or {})
        self._intervals: List[LabelInterval] = []
        self._load()

    def _map_person_id(self, label_person_id: int) -> int:
        return int(self._person_id_map.get(label_person_id, label_person_id))

    @classmethod
    def _parse_label(cls, label_text: str) -> Tuple[int, Optional[int]]:
        match = cls.LABEL_PATTERN.match(str(label_text).strip())
        if not match:
            raise ValueError(f"Invalid label_type '{label_text}'")
        source_person_id = int(match.group("src"))
        dst_group = match.group("dst")
        #if it is robot then dst is never defined 
        # since the right of the OR in the regex is evaluated
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

            intervals: List[LabelInterval] = []
            for row in reader:
                start_s = float(row["start_time"])
                end_s = float(row["end_time"])
                if end_s < start_s:
                    raise ValueError(f"Invalid interval with end_time < start_time: {row}")
                source_person_id, target_person_id = self._parse_label(row["label_type"])
                if target_person_id is None:
                    # Ignore robot-directed labels
                    continue
                source_person_id = self._map_person_id(source_person_id)
                target_person_id = self._map_person_id(target_person_id)
                intervals.append(
                    LabelInterval(
                        start_s=start_s,
                        end_s=end_s,
                        source_person_id=source_person_id,
                        target_person_id=target_person_id,
                    )
                )

            self._intervals = intervals

    def _expected_pairs_for_timestamp(self, timestamp_s: float) -> Dict[int, InteractionPair]:
        # One expected target per source at any given time; first interval encountered wins.
        expected_by_src: Dict[int, InteractionPair] = {}
        for interval in self._intervals:
            if interval.start_s <= timestamp_s < interval.end_s:
                src = interval.source_person_id
                expected_by_src.setdefault(src, (src, interval.target_person_id))
        return expected_by_src

    def compare_update(self, update) -> List[MismatchRecord]:
        timestamp_s = float(int(update.playback_timestamp_ns)) / 1e9
        expected_by_src = self._expected_pairs_for_timestamp(timestamp_s)

        mismatches: List[MismatchRecord] = []
        for edge in update.interactions:
            src = int(edge.from_person_id)
            dst = int(edge.to_person_id)
            is_looking = edge.is_looking

            pair = (src, dst)
            expected_pair = expected_by_src.get(src)
            if is_looking == (pair == expected_pair):
                continue

            actual_pair = pair if is_looking else None
            mismatches.append((actual_pair, expected_pair))

        return mismatches


# (is_looking_actual, is_expected) → human-readable kind label
_STATE_LABEL: Dict[Tuple[bool, bool], str] = {
    (True, False): "false_positive",   # looking, but no expected target for src
    (True, True): "wrong_target",      # looking at someone, but expected someone else
    (False, True): "false_negative",   # this is the expected pair, but not looking
}


@dataclass
class _ActiveMismatch:
    states: Set[Tuple[bool, bool]]  # set of (is_looking_actual, is_expected) tuples observed
    start_ts_ns: int
    start_frame: int
    last_ts_ns: int
    last_frame: int


class UniqueMismatchLogger:
    """Appends one JSONL entry per mismatch event (start..end).

    A mismatch event is a contiguous run of frames where a given edge
    (from_person_id, to_person_id) disagrees with ground truth in any way.
    The event ends only when the pair stops being mismatched (or the stream
    closes); state flips within the run extend the same event and are
    summarized via the `kinds` list. Every event is recorded regardless of
    duration; filtering by duration is a consumer concern (see replay).
    """

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        #create the parent directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.output_path.open("w", encoding="utf-8")


        self._activeMismatches: Dict[Tuple[int, int], _ActiveMismatch] = {}

    def close(self) -> None:
        try:
            for pair in list(self._activeMismatches.keys()):
                self._flush(pair)
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def handle_update(
        self,
        update: "rapport_pb2.PipelineUpdate",
        ground_truth: GroundTruthGazeComparator,
    ) -> None:
        ts_ns = update.playback_timestamp_ns
        frame_index = update.frame_index
        mismatches = ground_truth.compare_update(update)

        #(false,false) never happens because of the way the ground truth is constructed
        current_states: Dict[Tuple[int, int], Tuple[bool, bool]] = {}
        for actual_pair, expected_pair in mismatches:
            pair = actual_pair if actual_pair is not None else expected_pair
            current_states[pair] = (actual_pair is not None, expected_pair is not None)


        # flush any active mismatches that are no longer present in the current states
        for pair in list(self._activeMismatches.keys()):
            if pair not in current_states:
                self._flush(pair)

        # update any active mismatches that are still present in the current states
        for pair, state in current_states.items():
            active = self._activeMismatches.get(pair)
            if active is None:
                self._activeMismatches[pair] = _ActiveMismatch(
                    states={state},
                    start_ts_ns=ts_ns,
                    start_frame=frame_index,
                    last_ts_ns=ts_ns,
                    last_frame=frame_index,
                )
            else:
                active.states.add(state)
                active.last_ts_ns = ts_ns
                active.last_frame = frame_index

    def _flush(self, pair: Tuple[int, int]) -> None:
        active = self._activeMismatches.pop(pair, None)
        if active is None:
            return
        labels = sorted({_STATE_LABEL[state] for state in active.states})
        entry = {
            "from": pair[0],
            "to": pair[1],
            "start_playback_timestamp_ns": active.start_ts_ns,
            "end_playback_timestamp_ns": active.last_ts_ns,
            "duration_ns": max(0, active.last_ts_ns - active.start_ts_ns),
            "start_frame_index": active.start_frame,
            "end_frame_index": active.last_frame,
            "kinds": labels,
        }
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()
