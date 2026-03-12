"""
Speech emotion classification using superb/hubert-large-superb-er.
HuBERT-Large fine-tuned for IEMOCAP; 4 classes: angry, happy, neutral, sad.

Usage (from lab3/):
    python classify_emotion_hubert.py Toronto_samples emotion_results_hubert.csv
"""
import torch
from transformers import pipeline
import soundfile as sf
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# IEMOCAP 4-class mapping (in case pipeline returns LABEL_K)
# Order may vary by config; fallback to raw label if already a name
HUBERT_LABEL_MAP = {
    "LABEL_0": "angry",
    "LABEL_1": "happy",
    "LABEL_2": "neutral",
    "LABEL_3": "sad",
}


def classify_emotions(audio_folder, output_file):
    """
    Classify emotions in audio files using superb/hubert-large-superb-er.

    Args:
        audio_folder (str): Path to folder containing WAV files
        output_file (str): Path to save the CSV results
    """
    classifier = pipeline(
        "audio-classification",
        model="superb/hubert-large-superb-er",
        device=0 if torch.cuda.is_available() else -1,
    )

    audio_folder = Path(audio_folder)
    audio_files = list(audio_folder.glob("*.wav"))
    results = []

    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            speech, sample_rate = sf.read(str(audio_path))
            emotion_output = classifier(speech)

            top_emotion = emotion_output[0]
            raw_label = top_emotion["label"]
            emotion_name = HUBERT_LABEL_MAP.get(raw_label, raw_label)

            result = {
                "filename": audio_path.name,
                "emotion": emotion_name,
                "confidence": top_emotion["score"],
            }

            for item in emotion_output:
                key = f"{HUBERT_LABEL_MAP.get(item['label'], item['label'])}_probability"
                result[key] = item["score"]

            results.append(result)
            print(
                f"\nPredicted emotion for {audio_path.name}: {emotion_name} "
                f"(confidence: {top_emotion['score']:.2f})"
            )

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(results)} files")
    else:
        print("No results to save!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify emotions in WAV files (superb/hubert-large-superb-er)"
    )
    parser.add_argument("audio_folder", help="Folder containing WAV files")
    parser.add_argument("output_file", help="Path to save the CSV results")
    args = parser.parse_args()

    classify_emotions(args.audio_folder, args.output_file)
