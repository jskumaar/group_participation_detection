from __future__ import annotations

import argparse
import json
from pathlib import Path

from rapport_server import serve_forever


def cmd_serve(args: argparse.Namespace) -> int:
    serve_forever(
        host=args.host,
        port=args.port,
        ground_truth_csv=args.ground_truth_csv,
        person_id_map=args.person_id_map,
        mismatch_log_dir=args.mismatch_log_dir,
    )
    return 0


def _iter_jsonl_entries(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt JSON at line {line_number} in {path}: {exc}") from exc


def cmd_replay(args: argparse.Namespace) -> int:
    path = Path(args.log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    window_ns = int(max(0.0, float(args.mismatch_window)) * 1e9)

    total = 0
    shown = 0
    for entry in _iter_jsonl_entries(path):
        total += 1
        duration_ns = int(entry.get("duration_ns", 0))
        if duration_ns < window_ns:
            continue

        start_s = int(entry.get("start_source_timestamp_ns", 0)) / 1e9
        end_s = int(entry.get("end_source_timestamp_ns", 0)) / 1e9
        duration_s = duration_ns / 1e9
        print(
            "t={start:.3f}s-{end:.3f}s (dur={dur:.3f}s) frames={fs}-{fe} "
            "edge {src}->{dst} angle_deg={angle:.2f} expected={expected}".format(
                start=start_s,
                end=end_s,
                dur=duration_s,
                fs=entry.get("start_frame_index", -1),
                fe=entry.get("end_frame_index", -1),
                src=entry.get("from", -1),
                dst=entry.get("to", -1),
                angle=float(entry.get("angle_deg", 0.0)),
                expected=bool(entry.get("expected", False)),
            )
        )
        shown += 1

    if total == 0:
        print(f"No mismatch entries in log: {path}")
    else:
        print(
            f"\nShowing {shown} of {total} mismatch events "
            f"(duration >= {args.mismatch_window}s)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rapport bridge CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run Python gRPC analyzer server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=50051)
    serve_parser.add_argument(
        "--ground-truth-csv",
        help=(
            "CSV/TSV with columns: start_time, end_time, label_type "
            "(e.g. P1GazeP2, P3GazeRobot). Required to log mismatches."
        ),
    )
    serve_parser.add_argument(
        "--person-id-map",
        help=(
            "Optional mapping from label IDs to stream person_id values, "
            "format: 1:0,2:1,3:2"
        ),
    )
    serve_parser.add_argument(
        "--mismatch-log-dir",
        default="logs/mismatch_logs",
        help="Output directory for per-stream JSONL mismatch logs (report_N.jsonl)",
    )
    serve_parser.set_defaults(func=cmd_serve)

    replay_parser = subparsers.add_parser(
        "replay",
        help="Read and print a JSONL unique-mismatch log (report_N.jsonl)",
    )
    replay_parser.add_argument(
        "--log-path",
        required=True,
        help="Path to a JSONL mismatch log (e.g. logs/mismatch_logs/report_1.jsonl)",
    )
    replay_parser.add_argument(
        "--mismatch-window",
        type=float,
        default=2.0,
        help="Only show mismatch events whose duration (seconds) is >= this value (default 2.0)",
    )
    replay_parser.set_defaults(func=cmd_replay)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
