"""
Run all emotion classification models in this repository.

Each model script is invoked with: audio_folder, output_file
  python classify_emotion_<name>.py <audio_folder> <output_file>

Usage (from lab3/ or repo root):
  python run_all_emotion_models.py [audio_folder] [--output-dir DIR]

Examples:
  python run_all_emotion_models.py
  python run_all_emotion_models.py Toronto_samples
  python run_all_emotion_models.py Toronto_samples --output-dir results
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Script name (no path) -> default output filename (no path)
EMOTION_SCRIPTS = [
    ("classify_emotion_aventiq.py", "emotion_results_aventiq.csv"),
    ("classify_emotion_dpngtm.py", "emotion_results_dpngtm.csv"),
    ("classify_emotion_ehcalabres.py", "emotion_results_ehcalabres.csv"),
    ("classify_emotion_hubert.py", "emotion_results_hubert.csv"),
    ("classify_emotion_prithiv.py", "emotion_results_prithiv.csv"),
    ("classify_emotion_rf.py", "emotion_results_rf.csv"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all emotion classification models on a folder of WAV files."
    )
    parser.add_argument(
        "audio_folder",
        nargs="?",
        default="Toronto_samples",
        help="Folder containing WAV files (default: Toronto_samples)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory for output CSV files (default: current directory)",
    )
    args = parser.parse_args()

    lab3_dir = Path(__file__).resolve().parent
    audio_folder = Path(args.audio_folder)
    if not audio_folder.is_absolute():
        audio_folder = (lab3_dir / audio_folder).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (lab3_dir / output_dir).resolve()

    if not audio_folder.is_dir():
        print(f"Error: audio folder not found: {audio_folder}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    failed = []

    for script_name, out_csv in EMOTION_SCRIPTS:
        script_path = lab3_dir / script_name
        if not script_path.is_file():
            print(f"Skipping (not found): {script_name}")
            failed.append((script_name, "file not found"))
            continue

        output_file = output_dir / out_csv
        cmd = [python, str(script_path), str(audio_folder), str(output_file)]
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print("=" * 60)

        result = subprocess.run(cmd, cwd=str(lab3_dir))
        if result.returncode != 0:
            failed.append((script_name, f"exit code {result.returncode}"))

    print("\n" + "=" * 60)
    if failed:
        print("Failed runs:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
        sys.exit(1)
    print("All emotion models completed successfully.")


if __name__ == "__main__":
    main()
