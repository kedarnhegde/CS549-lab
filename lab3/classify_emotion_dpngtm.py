"""
Speech emotion classification using Dpngtm/wav2vec2-emotion-recognition.

This model is commonly trained on multiple SER datasets (e.g., TESS/RAVDESS/etc.).
It expects 16 kHz mono audio; we load and resample accordingly.

Usage (from lab3/):
  python classify_emotion_dpngtm.py Toronto_samples emotion_results_dpngtm.csv
"""

from __future__ import annotations

from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForAudioClassification, Wav2Vec2Processor


def _load_audio_16k_mono(audio_path: Path) -> torch.Tensor:
    """
    Load audio and return a 1D float tensor at 16kHz mono.
    Uses soundfile (same as other lab scripts); resamples with scipy if needed.
    """
    import soundfile as sf
    from scipy.signal import resample

    speech, sr = sf.read(str(audio_path), dtype="float32")
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    if sr != 16000:
        n = int(len(speech) * 16000 / sr)
        speech = resample(speech, n).astype("float32")
    return torch.tensor(speech, dtype=torch.float32)


def classify_emotions(audio_folder: str, output_file: str) -> None:
    model_id = "Dpngtm/wav2vec2-emotion-recognition"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForAudioClassification.from_pretrained(model_id).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model.eval()

    audio_folder_path = Path(audio_folder)
    audio_files = list(audio_folder_path.glob("*.wav"))
    results: list[dict] = []

    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            speech = _load_audio_16k_mono(audio_path).to(device)
            inputs = processor(
                speech,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0)

            pred_id = int(torch.argmax(probs).item())
            id2label = getattr(model.config, "id2label", {})
            pred_label = id2label.get(pred_id, f"LABEL_{pred_id}")
            pred_score = float(probs[pred_id].item())

            row = {
                "filename": audio_path.name,
                "emotion": str(pred_label),
                "confidence": pred_score,
            }

            # Add per-label probabilities
            for i in range(int(probs.numel())):
                label = id2label.get(i, f"LABEL_{i}")
                row[f"{str(label)}_probability"] = float(probs[i].item())

            results.append(row)
            print(
                f"\nPredicted emotion for {audio_path.name}: {pred_label} "
                f"(confidence: {pred_score:.2f})"
            )
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    if results:
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(results)} files")
    else:
        print("No results to save!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify emotions in WAV files (Dpngtm/wav2vec2-emotion-recognition)"
    )
    parser.add_argument("audio_folder", help="Folder containing WAV files")
    parser.add_argument("output_file", help="Path to save the CSV results")
    args = parser.parse_args()

    classify_emotions(args.audio_folder, args.output_file)

