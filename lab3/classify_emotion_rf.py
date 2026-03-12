"""
Speech emotion classification using r-f/wav2vec-english-speech-emotion-recognition.
Trained on TESS, RAVDESS, and SAVEE; 7 emotions. Good candidate for TESS-style data.

Usage (from lab3/):
    python classify_emotion_rf.py Toronto_samples emotion_results_rf.csv
"""
import torch
from transformers import pipeline
import soundfile as sf
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 7 emotions (from model card); map LABEL_K if pipeline returns raw labels
RF_LABEL_MAP = {
    "LABEL_0": "angry",
    "LABEL_1": "disgust",
    "LABEL_2": "fear",
    "LABEL_3": "happy",
    "LABEL_4": "neutral",
    "LABEL_5": "sad",
    "LABEL_6": "surprise",
}


def classify_emotions(audio_folder, output_file):
    """
    Classify emotions in audio files using r-f wav2vec2 emotion model.

    Args:
        audio_folder (str): Path to folder containing WAV files
        output_file (str): Path to save the CSV results
    """
    classifier = pipeline(
        "audio-classification",
        model="r-f/wav2vec-english-speech-emotion-recognition",
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
            emotion_name = RF_LABEL_MAP.get(raw_label, raw_label)

            result = {
                "filename": audio_path.name,
                "emotion": emotion_name,
                "confidence": top_emotion["score"],
            }

            for item in emotion_output:
                key = f"{RF_LABEL_MAP.get(item['label'], item['label'])}_probability"
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
        description="Classify emotions in WAV files (r-f/wav2vec-english-speech-emotion-recognition)"
    )
    parser.add_argument("audio_folder", help="Folder containing WAV files")
    parser.add_argument("output_file", help="Path to save the CSV results")
    args = parser.parse_args()

    classify_emotions(args.audio_folder, args.output_file)
