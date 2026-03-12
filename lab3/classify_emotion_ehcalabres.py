"""
Speech emotion classification using ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition.

Usage (from lab3/):
    python classify_emotion_ehcalabres.py Toronto_samples emotion_results_ehcalabres.csv
"""
import torch
from transformers import pipeline
import soundfile as sf
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def classify_emotions(audio_folder, output_file):
    """
    Classify emotions in audio files using a pre-trained model.
    
    Args:
        audio_folder (str): Path to folder containing audio files
        output_file (str): Path to save the CSV results
    """
    # Initialize the emotion recognition pipeline
    classifier = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Get all wav files in the folder
    audio_folder = Path(audio_folder)
    audio_files = list(audio_folder.glob("*.wav"))
    
    results = []
    
    # Process each audio file
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load audio file
            speech, sample_rate = sf.read(str(audio_path))
            
            # Classify emotion
            emotion_output = classifier(speech)
            
            # Get top prediction
            top_emotion = emotion_output[0]
            
            # Store results
            result = {
                'filename': audio_path.name,
                'emotion': top_emotion['label'],
                'confidence': top_emotion['score'],
            }
            
            # Add all emotion probabilities
            for item in emotion_output:
                result[f"{item['label']}_probability"] = item['score']
            
            results.append(result)
            print(f"\nPredicted emotion for {audio_path.name}: {top_emotion['label']} "
                  f"(confidence: {top_emotion['score']:.2f})")
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(results)} files")
    else:
        print("No results to save!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify emotions in audio files")
    parser.add_argument("audio_folder", help="Folder containing WAV files")
    parser.add_argument("output_file", help="Path to save the CSV results")
    
    args = parser.parse_args()
    
    classify_emotions(args.audio_folder, args.output_file)