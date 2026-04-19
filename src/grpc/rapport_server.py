from __future__ import annotations

import json
import re
import sys
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, TextIO, Tuple

import grpc

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
if str(GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATED_DIR))

import rapport_pb2
import rapport_pb2_grpc
from gaze_ground_truth import GroundTruthGazeComparator


def parse_person_id_map(mapping_text: Optional[str]) -> Dict[int, int]:
    if mapping_text is None or not mapping_text.strip():
        return {}
    result: Dict[int, int] = {}
    for item in mapping_text.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid mapping entry '{part}'. Expected format source:target")
        src_text, dst_text = part.split(":", 1)
        src = int(src_text.strip())
        dst = int(dst_text.strip())
        result[src] = dst
    return result


def _next_report_path(output_dir: Path, suffix: str) -> Path:
    pattern = re.compile(rf"^report_(\d+)\.{re.escape(suffix)}$")
    max_index = 0
    if output_dir.exists():
        for existing in output_dir.iterdir():
            if not existing.is_file():
                continue
            match = pattern.match(existing.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return output_dir / f"report_{max_index + 1}.{suffix}"


@dataclass
class _ActiveMismatch:
    state: Tuple[bool, bool]  # (is_looking_actual, expected)
    start_ts_ns: int
    start_frame: int
    last_ts_ns: int
    last_frame: int
    first_angle_deg: float


class UniqueMismatchLogger:
    """Appends one JSONL entry per unique mismatch event (start..end).

    A mismatch event is a contiguous run of frames where a given edge
    (from_person_id, to_person_id) has the same (is_looking_actual, expected)
    tuple. An entry is written when that run ends (edge no longer mismatched,
    state flips, or stream closes). Every such event is recorded regardless
    of duration; filtering by duration is a consumer concern (see replay).
    """

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.output_path.open("w", encoding="utf-8")
        self._active: Dict[Tuple[int, int], _ActiveMismatch] = {}

    def close(self) -> None:
        try:
            for pair in list(self._active.keys()):
                self._flush(pair)
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def handle_update(
        self,
        update: "rapport_pb2.PipelineUpdate",
        ground_truth: GroundTruthGazeComparator,
    ) -> int:
        ts_ns = int(getattr(update, "source_timestamp_ns", 0))
        frame_index = int(getattr(update, "frame_index", 0))
        timestamp_s = ts_ns / 1e9
        expected_by_pair = ground_truth._expected_pairs_for_timestamp(timestamp_s)

        actual_by_pair: Dict[Tuple[int, int], Tuple[bool, float]] = {}
        for edge in update.interactions:
            key = (int(edge.from_person_id), int(edge.to_person_id))
            actual_by_pair[key] = (
                bool(getattr(edge, "is_looking", False)),
                float(getattr(edge, "angle_deg", 0.0)),
            )

        current_states: Dict[Tuple[int, int], Tuple[Tuple[bool, bool], float]] = {}
        for pair, expected_value in expected_by_pair.items():
            actual_tuple = actual_by_pair.get(pair)
            if actual_tuple is None:
                continue
            actual_looking, angle_deg = actual_tuple
            if actual_looking == bool(expected_value):
                continue
            state = (actual_looking, bool(expected_value))
            current_states[pair] = (state, angle_deg)

        entries_written = 0

        for pair in list(self._active.keys()):
            if pair not in current_states or current_states[pair][0] != self._active[pair].state:
                self._flush(pair)
                entries_written += 1

        for pair, (state, angle_deg) in current_states.items():
            active = self._active.get(pair)
            if active is None:
                self._active[pair] = _ActiveMismatch(
                    state=state,
                    start_ts_ns=ts_ns,
                    start_frame=frame_index,
                    last_ts_ns=ts_ns,
                    last_frame=frame_index,
                    first_angle_deg=angle_deg,
                )
            else:
                active.last_ts_ns = ts_ns
                active.last_frame = frame_index

        return entries_written

    def _flush(self, pair: Tuple[int, int]) -> None:
        active = self._active.pop(pair, None)
        if active is None:
            return
        entry = {
            "from": pair[0],
            "to": pair[1],
            "start_source_timestamp_ns": active.start_ts_ns,
            "end_source_timestamp_ns": active.last_ts_ns,
            "duration_ns": max(0, active.last_ts_ns - active.start_ts_ns),
            "start_frame_index": active.start_frame,
            "end_frame_index": active.last_frame,
            "angle_deg": round(float(active.first_angle_deg), 3),
            "is_looking": bool(active.state[0]),
            "expected": bool(active.state[1]),
        }
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()


class RapportStreamServicer(rapport_pb2_grpc.RapportStreamServicer):
    def __init__(
        self,
        ground_truth: Optional[GroundTruthGazeComparator] = None,
        mismatch_log_dir: str = "logs/mismatch_logs",
    ) -> None:
        self._ground_truth = ground_truth
        self._mismatch_log_dir = Path(mismatch_log_dir)

    def StreamPipelineUpdates(self, request_iterator, context):
        count = 0
        mismatch_logger: Optional[UniqueMismatchLogger] = None
        if self._ground_truth is not None:
            log_path = _next_report_path(self._mismatch_log_dir, "jsonl")
            mismatch_logger = UniqueMismatchLogger(log_path)
            print(f"Logging unique mismatches to {log_path}")

        try:
            for update in request_iterator:
                if mismatch_logger is not None and self._ground_truth is not None:
                    mismatch_logger.handle_update(update, self._ground_truth)
                count += 1
        finally:
            if mismatch_logger is not None:
                mismatch_logger.close()
                print(f"Closed mismatch log {mismatch_logger.output_path}")

        return rapport_pb2.StreamAck(
            received_messages=count,
            message=f"Buffered {count} updates",
        )


def _build_server(
    host: str,
    port: int,
    ground_truth_csv: Optional[str],
    person_id_map: Optional[str],
    mismatch_log_dir: str,
) -> grpc.Server:
    id_map = parse_person_id_map(person_id_map)
    ground_truth = (
        GroundTruthGazeComparator(ground_truth_csv, person_id_map=id_map)
        if ground_truth_csv
        else None
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    rapport_pb2_grpc.add_RapportStreamServicer_to_server(
        RapportStreamServicer(
            ground_truth=ground_truth,
            mismatch_log_dir=mismatch_log_dir,
        ),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server


def serve(
    host: str = "127.0.0.1",
    port: int = 50051,
    ground_truth_csv: Optional[str] = None,
    person_id_map: Optional[str] = None,
    mismatch_log_dir: str = "logs/mismatch_logs",
) -> grpc.Server:
    return _build_server(
        host=host,
        port=port,
        ground_truth_csv=ground_truth_csv,
        person_id_map=person_id_map,
        mismatch_log_dir=mismatch_log_dir,
    )


def serve_forever(
    host: str = "127.0.0.1",
    port: int = 50051,
    ground_truth_csv: Optional[str] = None,
    person_id_map: Optional[str] = None,
    mismatch_log_dir: str = "logs/mismatch_logs",
) -> None:
    server = _build_server(
        host=host,
        port=port,
        ground_truth_csv=ground_truth_csv,
        person_id_map=person_id_map,
        mismatch_log_dir=mismatch_log_dir,
    )
    print(f"Rapport gRPC server listening on {host}:{port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(grace=2)
