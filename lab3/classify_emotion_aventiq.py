"""
Speech emotion classification using AventIQ-AI/wav2vec2-base_speech_emotion_recognition.
Trained on RAVDESS; maps model output labels to readable emotion names.
"""
import torch
from transformers import pipeline
import soundfile as sf
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Map LABEL_K to emotion names (RAVDESS 8 classes)
AVENTIQ_LABEL_MAP = {
    "LABEL_0": "neutral",
    "LABEL_1": "calm",
    "LABEL_2": "happy",
    "LABEL_3": "sad",
    "LABEL_4": "angry",
    "LABEL_5": "fearful",
    "LABEL_6": "disgust",
    "LABEL_7": "surprised",
}


def classify_emotions(audio_folder, output_file):
    """
    Classify emotions in audio files using AventIQ-AI wav2vec2 emotion model.

    Args:
        audio_folder (str): Path to folder containing WAV files
        output_file (str): Path to save the CSV results
    """
    classifier = pipeline(
        "audio-classification",
        model="AventIQ-AI/wav2vec2-base_speech_emotion_recognition",
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
            emotion_name = AVENTIQ_LABEL_MAP.get(raw_label, raw_label)

            result = {
                "filename": audio_path.name,
                "emotion": emotion_name,
                "confidence": top_emotion["score"],
            }

            for item in emotion_output:
                key = f"{AVENTIQ_LABEL_MAP.get(item['label'], item['label'])}_probability"
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
        description="Classify emotions in WAV files (AventIQ-AI model)"
    )
    parser.add_argument("audio_folder", help="Folder containing WAV files")
    parser.add_argument("output_file", help="Path to save the CSV results")
    args = parser.parse_args()

    classify_emotions(args.audio_folder, args.output_file)
