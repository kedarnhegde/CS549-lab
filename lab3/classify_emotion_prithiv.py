"""
Speech emotion classification using prithivMLmods/Speech-Emotion-Classification.
8 classes: Anger, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprised.

Usage (from lab3/):
    python classify_emotion_prithiv.py Toronto_samples emotion_results_prithivMLmods.csv
"""
import torch
from transformers import pipeline
import soundfile as sf
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Map LABEL_K to emotion names (8 classes)
PRITHIV_LABEL_MAP = {
    "LABEL_0": "Anger",
    "LABEL_1": "Calm",
    "LABEL_2": "Disgust",
    "LABEL_3": "Fear",
    "LABEL_4": "Happy",
    "LABEL_5": "Neutral",
    "LABEL_6": "Sad",
    "LABEL_7": "Surprised",
}


def classify_emotions(audio_folder, output_file):
    """
    Classify emotions in audio files using prithivMLmods Speech-Emotion-Classification.

    Args:
        audio_folder (str): Path to folder containing WAV files
        output_file (str): Path to save the CSV results
    """
    classifier = pipeline(
        "audio-classification",
        model="prithivMLmods/Speech-Emotion-Classification",
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
            emotion_name = PRITHIV_LABEL_MAP.get(raw_label, raw_label)

            result = {
                "filename": audio_path.name,
                "emotion": emotion_name,
                "confidence": top_emotion["score"],
            }

            for item in emotion_output:
                key = f"{PRITHIV_LABEL_MAP.get(item['label'], item['label'])}_probability"
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
        description="Classify emotions in WAV files (prithivMLmods/Speech-Emotion-Classification)"
    )
    parser.add_argument("audio_folder", help="Folder containing WAV files")
    parser.add_argument("output_file", help="Path to save the CSV results")
    args = parser.parse_args()

    classify_emotions(args.audio_folder, args.output_file)
