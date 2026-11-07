from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from extract_labels import GroundTruthGazeComparator

NO_PERSON = "no_person"
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _target_name(target: Optional[int]) -> str:
    return NO_PERSON if target is None else f"P{target}"


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator) / float(denominator) if denominator else None


def _precision_recall_f1(predicted: np.ndarray, truth: np.ndarray) -> Dict[str, Optional[float]]:
    true_positives = int(np.sum(predicted & truth))
    precision = _ratio(true_positives, int(np.sum(predicted)))
    recall = _ratio(true_positives, int(np.sum(truth)))
    f1 = (
        _ratio(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _auroc(probabilities: np.ndarray, truth: np.ndarray) -> Optional[float]:
    positives = int(np.sum(truth))
    negatives = truth.size - positives
    if positives == 0 or negatives == 0:
        return None
    # Mann-Whitney U with average ranks, so tied probabilities count as half.
    _, inverse, counts = np.unique(probabilities, return_inverse=True, return_counts=True)
    first_rank = np.concatenate(([0], np.cumsum(counts)[:-1])) + 1
    ranks = (first_rank + (counts - 1) / 2.0)[inverse]
    return float((np.sum(ranks[truth]) - positives * (positives + 1) / 2.0) / (positives * negatives))


def _average_precision(probabilities: np.ndarray, truth: np.ndarray) -> Optional[float]:
    positives = int(np.sum(truth))
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    true_positives = np.cumsum(truth[order])
    # Evaluate only where the threshold changes, so ties form a single operating point.
    is_cutoff = np.append(sorted_probabilities[1:] != sorted_probabilities[:-1], True)
    tp_at_cutoff = true_positives[is_cutoff]
    predicted_at_cutoff = np.flatnonzero(is_cutoff) + 1
    precision = tp_at_cutoff / predicted_at_cutoff
    recall_gain = np.diff(np.concatenate(([0], tp_at_cutoff))) / positives
    return float(np.sum(precision * recall_gain))


class GazeEvaluator:
    """Scores the model against ground truth over one stream.

    Two views of the same frames:
      - Per person: each labeled person gets one prediction per frame (a target
        person or no_person), scored against the accepted targets. This is the
        accuracy and confusion matrix. Robot labels are the no_person class.
      - Per edge: each observer -> target probability is scored as a binary
        forecast of "is this person looking at that person". This judges the
        posterior itself and shows what a threshold other than argmax would do.
    """

    def __init__(self, ground_truth: GroundTruthGazeComparator) -> None:
        self.ground_truth = ground_truth
        self.frames_received = 0
        self.frame_gaps = 0
        self._last_frame_index: Optional[int] = None
        self._last_timestamp_s = 0.0

        self._scored: Counter = Counter()
        self._correct: Counter = Counter()
        self._no_person_accepted: Counter = Counter()
        self._no_prediction: Counter = Counter()
        self._unlabeled: Counter = Counter()
        self._confusion: Dict[str, Counter] = defaultdict(Counter)

        self._edge_probabilities: List[float] = []
        self._edge_truth: List[bool] = []
        self._edge_argmax: List[bool] = []

    def handle_update(self, update) -> None:
        self.frames_received += 1
        frame_index = int(update.frame_index)
        if self._last_frame_index is not None and frame_index != self._last_frame_index + 1:
            self.frame_gaps += 1
        self._last_frame_index = frame_index

        timestamp_s = update.playback_timestamp_ns / 1e9
        self._last_timestamp_s = timestamp_s
        if timestamp_s >= self.ground_truth.labels_end_s:
            return

        edges_by_src: Dict[int, list] = defaultdict(list)
        for edge in update.interactions:
            edges_by_src[int(edge.from_person_id)].append(edge)

        accepted_by_src = self.ground_truth.accepted_targets(timestamp_s)
        for src in self.ground_truth.sources:
            accepted = accepted_by_src.get(src)
            if accepted is None:
                self._unlabeled[src] += 1
                continue

            edges = edges_by_src.get(src)
            if not edges:
                self._no_prediction[src] += 1
                continue

            predicted = next((int(e.to_person_id) for e in edges if e.is_looking), None)
            is_correct = predicted in accepted
            self._scored[src] += 1
            self._correct[src] += int(is_correct)
            self._no_person_accepted[src] += int(None in accepted)

            # Under overlapping labels, credit the label the model actually chose.
            expected = predicted if is_correct else min(
                accepted, key=lambda target: (target is None, target)
            )
            self._confusion[_target_name(expected)][_target_name(predicted)] += 1

            for edge in edges:
                self._edge_probabilities.append(float(edge.probability))
                self._edge_truth.append(int(edge.to_person_id) in accepted)
                self._edge_argmax.append(bool(edge.is_looking))

    def progress_line(self) -> str:
        scored = sum(self._scored.values())
        accuracy = _ratio(sum(self._correct.values()), scored)
        accuracy_text = "n/a" if accuracy is None else f"{accuracy:.3f}"
        return (
            f"t={self._last_timestamp_s:.1f}s frames={self.frames_received} "
            f"scored_predictions={scored} accuracy={accuracy_text}"
        )

    def _edge_metrics(self) -> Optional[dict]:
        if not self._edge_probabilities:
            return None
        probabilities = np.asarray(self._edge_probabilities, dtype=np.float64)
        truth = np.asarray(self._edge_truth, dtype=bool)
        argmax = np.asarray(self._edge_argmax, dtype=bool)
        return {
            "count": int(truth.size),
            "positive_rate": float(np.mean(truth)),
            "auroc": _auroc(probabilities, truth),
            "average_precision": _average_precision(probabilities, truth),
            "brier": float(np.mean((probabilities - truth) ** 2)),
            "max_probability": float(np.max(probabilities)),
            "argmax": _precision_recall_f1(argmax, truth),
            "thresholds": [
                {"threshold": t, **_precision_recall_f1(probabilities >= t, truth)}
                for t in THRESHOLDS
            ],
        }

    def summary(self) -> dict:
        scored = sum(self._scored.values())
        per_source = {}
        for src in sorted(self.ground_truth.sources):
            per_source[f"P{src}"] = {
                "scored": self._scored[src],
                "accuracy": _ratio(self._correct[src], self._scored[src]),
                "no_prediction": self._no_prediction[src],
                "unlabeled": self._unlabeled[src],
            }
        return {
            "frames_received": self.frames_received,
            "frame_gaps": self.frame_gaps,
            "last_timestamp_s": self._last_timestamp_s,
            "labels_end_s": self.ground_truth.labels_end_s,
            "scored_predictions": scored,
            "accuracy": _ratio(sum(self._correct.values()), scored),
            "always_no_person_accuracy": _ratio(sum(self._no_person_accepted.values()), scored),
            "per_source": per_source,
            "confusion": {
                expected: dict(predicted) for expected, predicted in sorted(self._confusion.items())
            },
            "edges": self._edge_metrics(),
        }

    def write_summary(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, indent=2)
            fh.write("\n")

    def format_summary(self) -> str:
        def number(value: Optional[float]) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        summary = self.summary()
        lines = [
            "",
            "=== Gaze evaluation ===",
            f"Frames received: {summary['frames_received']} "
            f"(video reached {summary['last_timestamp_s']:.1f}s, "
            f"labels end {summary['labels_end_s']:.1f}s)",
            f"Accuracy: {number(summary['accuracy'])} over {summary['scored_predictions']} "
            f"predictions (always predicting no_person scores "
            f"{number(summary['always_no_person_accuracy'])})",
        ]

        for name, stats in summary["per_source"].items():
            lines.append(
                f"  {name}: accuracy={number(stats['accuracy'])} scored={stats['scored']} "
                f"no_prediction={stats['no_prediction']} unlabeled={stats['unlabeled']}"
            )

        names = sorted({n for row in summary["confusion"].values() for n in row} | set(summary["confusion"]))
        if names:
            lines.extend(["", "Confusion (rows = labels, columns = predictions):"])
            lines.append(" " * 11 + "".join(f"{name:>11}" for name in names))
            for expected in names:
                row = summary["confusion"].get(expected, {})
                lines.append(f"{expected:<11}" + "".join(f"{row.get(name, 0):>11}" for name in names))

        edges = summary["edges"]
        if edges is not None:
            argmax = edges["argmax"]
            lines.extend(
                [
                    "",
                    f"Edge probability over {edges['count']} edges "
                    f"(base rate {edges['positive_rate']:.3f}):",
                    f"  AUROC={number(edges['auroc'])} "
                    f"AP={number(edges['average_precision'])} Brier={number(edges['brier'])}",
                    f"  argmax: precision={number(argmax['precision'])} "
                    f"recall={number(argmax['recall'])} f1={number(argmax['f1'])}",
                    "  threshold  precision  recall     f1",
                ]
            )
            for row in edges["thresholds"]:
                lines.append(
                    f"  {row['threshold']:<9.1f}  {number(row['precision']):<9}  "
                    f"{number(row['recall']):<9}  {number(row['f1'])}"
                )
            if edges["max_probability"] <= 0.0:
                lines.append(
                    "  WARNING: every probability is 0, so tracker.exe is older than rapport.proto. "
                    "Rebuild it."
                )

        if summary["frame_gaps"]:
            lines.extend(
                [
                    "",
                    f"WARNING: {summary['frame_gaps']} jumps in frame_index. Frames were dropped "
                    "(client joined late or fell behind) or the video was seeked, so these "
                    "numbers may be off.",
                ]
            )
        return "\n".join(lines)
