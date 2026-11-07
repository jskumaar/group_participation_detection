from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional
import time

import grpc
from google.protobuf import empty_pb2

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
if str(GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATED_DIR))

import rapport_pb2_grpc
from evaluation import GazeEvaluator
from extract_labels import GroundTruthGazeComparator, UniqueMismatchLogger

REPORT_SUFFIX = "jsonl"
PROGRESS_EVERY_FRAMES = 300


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


def next_report_path(output_dir: Path) -> Path:
    max_index = 0
    if output_dir.exists():
        for entry in output_dir.glob(f"report_*.{REPORT_SUFFIX}"):
            if not entry.is_file():
                continue
            index_text = entry.stem.removeprefix("report_")
            if index_text.isdigit():
                max_index = max(max_index, int(index_text))
    return output_dir / f"report_{max_index + 1}.{REPORT_SUFFIX}"

def _build_comparator(
    ground_truth_csv: Optional[str],
    person_id_map: Optional[str],
) -> Optional[GroundTruthGazeComparator]:
    id_map = parse_person_id_map(person_id_map)
    return (
        GroundTruthGazeComparator(ground_truth_csv, person_id_map=id_map)
        if ground_truth_csv
        else None
    )

def run_client_once(
    host: str = "127.0.0.1",
    port: int = 50051,
    ground_truth: Optional[GroundTruthGazeComparator] = None,
    mismatch_log_dir: str = "logs/mismatch_logs",
) -> None:
    endpoint = f"{host}:{port}"
    # Created on the first update so that waiting for the tracker leaves no empty reports.
    mismatch_logger: Optional[UniqueMismatchLogger] = None
    evaluator: Optional[GazeEvaluator] = None
    announced_labels_end = False

    try:
        with grpc.insecure_channel(endpoint) as channel:
            stub = rapport_pb2_grpc.RapportStreamStub(channel)
            # wait_for_ready holds the call open until the tracker is listening.
            updates = stub.StreamPipelineUpdates(empty_pb2.Empty(), wait_for_ready=True)
            print(f"Waiting for tracker updates on {endpoint} ...")
            for update in updates:
                if ground_truth is None:
                    continue

                if mismatch_logger is None:
                    log_path = next_report_path(Path(mismatch_log_dir))
                    mismatch_logger = UniqueMismatchLogger(log_path)
                    evaluator = GazeEvaluator(ground_truth)
                    print(f"Logging unique mismatches to {log_path}")

                mismatch_logger.handle_update(update, ground_truth)
                evaluator.handle_update(update)

                if evaluator.frames_received % PROGRESS_EVERY_FRAMES == 0:
                    print(evaluator.progress_line())

                timestamp_s = update.playback_timestamp_ns / 1e9
                if not announced_labels_end and timestamp_s >= ground_truth.labels_end_s:
                    announced_labels_end = True
                    print(
                        f"Passed the last label at {ground_truth.labels_end_s:.2f}s, "
                        "later frames are not scored. Press Ctrl+C for the final report."
                    )
    finally:
        if mismatch_logger is not None:
            mismatch_logger.close()
            print(f"Closed mismatch log {mismatch_logger.output_path}")
        if evaluator is not None:
            summary_path = mismatch_logger.output_path.with_name(
                f"{mismatch_logger.output_path.stem}_summary.json"
            )
            evaluator.write_summary(summary_path)
            print(evaluator.format_summary())
            print(f"Wrote summary to {summary_path}")


def serve_forever(
    host: str = "127.0.0.1",
    port: int = 50051,
    ground_truth_csv: Optional[str] = None,
    person_id_map: Optional[str] = None,
    mismatch_log_dir: str = "logs/mismatch_logs",
) -> None:
    # Outside the retry loop: a bad labels file should fail once, not every second.
    ground_truth = _build_comparator(
        ground_truth_csv=ground_truth_csv,
        person_id_map=person_id_map,
    )
    try:
        while True:
            try:
                run_client_once(
                    host=host,
                    port=port,
                    ground_truth=ground_truth,
                    mismatch_log_dir=mismatch_log_dir,
                )
            except grpc.RpcError as exc:
                print(f"gRPC stream error: {exc.code().name} - {exc.details()}")
                time.sleep(1.0)
            except Exception as exc:
                print(f"Client loop error: {exc}")
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
