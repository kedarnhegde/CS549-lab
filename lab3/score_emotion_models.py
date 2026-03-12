"""
Score speech-emotion model CSV outputs against TESS ground truth.

Assumes each CSV has at least:
  - filename
  - emotion  (model prediction label)

Ground truth is extracted from filename like: OAF_beg_happy.wav -> happy

Usage examples (run from lab3/):
  python score_emotion_models.py emotion_results_AventIQ.csv
  python score_emotion_models.py emotion_results_*.csv
  python score_emotion_models.py --glob "emotion_results_*.csv"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import glob
from typing import Iterable, Optional, Dict, List

import pandas as pd


def _normalize_emotion(label: object) -> str:
    """Normalize label strings across different model naming conventions."""
    s = str(label).strip().lower()
    mapping = {
        # common abbreviations / variants seen in some model outputs
        "ang": "angry",
        "anger": "angry",
        "hap": "happy",
        "happiness": "happy",
        "neu": "neutral",
        "fearful": "fear",
        "surprised": "surprise",
        "pleasant_surprise": "surprise",
    }
    return mapping.get(s, s)


def _extract_actual_from_filename(filename: str) -> str:
    # Example: OAF_beg_happy.wav -> happy
    stem = Path(filename).stem
    return stem.split("_")[-1]


@dataclass(frozen=True)
class ScoreResult:
    file: str
    n: int
    correct: int
    accuracy: float
    avg_confidence: Optional[float]
    macro_f1: Optional[float]


def _macro_f1(y_true: Iterable[str], y_pred: Iterable[str]) -> Optional[float]:
    try:
        from sklearn.metrics import f1_score  # type: ignore
    except ImportError:
        return None
    return float(f1_score(list(y_true), list(y_pred), average="macro", zero_division=0))


def score_file(
    csv_path: Path,
) -> tuple[
    ScoreResult,
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
    List[Dict[str, str]],
]:
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "emotion" not in df.columns:
        raise ValueError(f"{csv_path.name} must contain columns: filename, emotion")

    # Average confidence if the column is present
    if "confidence" in df.columns:
        avg_conf = float(df["confidence"].mean())
    else:
        avg_conf = None

    actual = df["filename"].astype(str).map(_extract_actual_from_filename).map(_normalize_emotion)
    pred = df["emotion"].map(_normalize_emotion)

    correct = int((pred == actual).sum())
    n = int(len(df))
    acc = correct / n if n else 0.0
    f1 = _macro_f1(actual, pred)

    # Per-emotion stats for this file
    per_emotion: Dict[str, Dict[str, int]] = {}
    # Confusion counts: actual -> predicted -> count
    confusion: Dict[str, Dict[str, int]] = {}
    # Misclassified examples for this file (for later error analysis)
    misclassified: List[Dict[str, str]] = []
    for idx, (a, p) in zip(df.index, zip(actual, pred)):
        if a not in per_emotion:
            per_emotion[a] = {"correct": 0, "total": 0}
        per_emotion[a]["total"] += 1
        if a == p:
            per_emotion[a]["correct"] += 1
        else:
            misclassified.append(
                {
                    "model_file": csv_path.name,
                    "filename": str(df.loc[idx, "filename"]),
                    "actual": a,
                    "pred": p,
                }
            )
        # confusion matrix counts
        if a not in confusion:
            confusion[a] = {}
        confusion[a][p] = confusion[a].get(p, 0) + 1

    return (
        ScoreResult(
            file=csv_path.name,
            n=n,
            correct=correct,
            accuracy=acc,
            avg_confidence=avg_conf,
            macro_f1=f1,
        ),
        per_emotion,
        confusion,
        misclassified,
    )


def _format_pct(x: float) -> str:
    return f"{x:.2%}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score emotion model CSV outputs.")
    parser.add_argument("csv", nargs="*", help="CSV file(s) to score")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help='Glob pattern, e.g. "emotion_results_*.csv"')
    args = parser.parse_args()

    paths: list[Path] = []
    if args.glob_pattern:
        paths.extend(sorted(Path(".").glob(args.glob_pattern)))

    # On Windows/PowerShell, wildcard arguments like emotion_results_*.csv are not
    # always expanded by the shell. Expand them here if they contain glob chars.
    for raw in args.csv:
        if any(ch in raw for ch in ("*", "?", "[")):
            expanded = glob.glob(raw)
            if expanded:
                paths.extend([Path(p) for p in expanded])
                continue
        paths.append(Path(raw))

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique_paths: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(p)

    if not unique_paths:
        # default behavior: score all emotion_results_*.csv in current dir
        unique_paths = sorted(Path(".").glob("emotion_results_*.csv"))

    results: list[ScoreResult] = []
    # Global per-emotion stats across all scored files
    global_per_emotion: Dict[str, Dict[str, int]] = {}
    # Global confusion counts across all scored files
    global_confusion: Dict[str, Dict[str, int]] = {}
    # All misclassified examples across all files
    all_misclassified: List[Dict[str, str]] = []

    for p in unique_paths:
        score, per_emotion, confusion, misclassified = score_file(p)
        results.append(score)
        for emo, stats in per_emotion.items():
            if emo not in global_per_emotion:
                global_per_emotion[emo] = {"correct": 0, "total": 0}
            global_per_emotion[emo]["correct"] += stats["correct"]
            global_per_emotion[emo]["total"] += stats["total"]
        for actual, row in confusion.items():
            if actual not in global_confusion:
                global_confusion[actual] = {}
            for pred_label, count in row.items():
                global_confusion[actual][pred_label] = (
                    global_confusion[actual].get(pred_label, 0) + count
                )
        all_misclassified.extend(misclassified)

    # Print a compact table
    print(f"{'file':40}  {'acc':>8}  {'avg_conf':>9}  {'correct':>9}  {'n':>4}  {'macro_f1':>8}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x.accuracy, reverse=True):
        f1 = "" if r.macro_f1 is None else f"{r.macro_f1:.4f}"
        avg_conf_str = "" if r.avg_confidence is None else f"{r.avg_confidence:.3f}"
        print(
            f"{r.file:40}  {_format_pct(r.accuracy):>8}  {avg_conf_str:>9}  "
            f"{r.correct:>9}  {r.n:>4}  {f1:>8}"
        )

    if any(r.macro_f1 is None for r in results):
        print("\nNote: macro_f1 is blank because scikit-learn is not installed.")
        print("Install it with: pip install scikit-learn")

    # Per-emotion summary across all models/files
    worst_emo = None
    best_emo = None
    worst_acc = None
    best_acc = None

    for emo, stats in global_per_emotion.items():
        if stats["total"] == 0:
            continue
        emo_acc = stats["correct"] / stats["total"]
        if worst_acc is None or emo_acc < worst_acc:
            worst_acc = emo_acc
            worst_emo = emo
        if best_acc is None or emo_acc > best_acc:
            best_acc = emo_acc
            best_emo = emo

    if worst_emo is not None and best_emo is not None:
        print("\nPer-emotion performance across all models:")
        # Helper to find most frequent confusion partner for a given emotion
        def _most_confused_with(emo: str) -> str:
            row = global_confusion.get(emo, {})
            # Exclude correct predictions (emo -> emo)
            candidates = {k: v for k, v in row.items() if k != emo}
            if not candidates:
                return "none (mostly predicted correctly)"
            return max(candidates.items(), key=lambda kv: kv[1])[0]

        worst_conf = _most_confused_with(worst_emo)
        best_conf = _most_confused_with(best_emo)

        print(
            f"  Worst: {worst_emo} "
            f"({worst_acc:.2%} correct, "
            f"{global_per_emotion[worst_emo]['correct']}/"
            f"{global_per_emotion[worst_emo]['total']}, "
            f'most confused with "{worst_conf}")'
        )
        print(
            f"  Best : {best_emo} "
            f"({best_acc:.2%} correct, "
            f"{global_per_emotion[best_emo]['correct']}/"
            f"{global_per_emotion[best_emo]['total']}, "
            f'most confused with "{best_conf}")'
        )

    # Show a few concrete misclassified examples for error analysis
    if all_misclassified:
        print("\nExample misclassified samples (up to 3):")
        for ex in all_misclassified[:3]:
            print(
                f"  model={ex['model_file']}, file={ex['filename']}, "
                f"true={ex['actual']}, pred={ex['pred']}"
            )

    # Show top confusion patterns (actual -> predicted with highest counts)
    confusion_pairs: List[tuple[str, str, int]] = []
    for actual, row in global_confusion.items():
        for pred_label, count in row.items():
            if actual == pred_label:
                continue
            confusion_pairs.append((actual, pred_label, count))
    if confusion_pairs:
        confusion_pairs.sort(key=lambda t: t[2], reverse=True)
        print("\nTop confusion patterns across all models:")
        for actual, pred_label, count in confusion_pairs[:3]:
            print(f"  {actual} → {pred_label}: {count} times")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

