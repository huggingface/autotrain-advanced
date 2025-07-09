import os
import base64
import pandas as pd
import tempfile


def convert_life_app_json_to_local_dataset(dataset_json):
    """
    Convert a LiFE App dataset JSON (list of dicts with base64-encoded audio and transcription)
    into a temporary folder with audio files and a CSV for training.
    Args:
        dataset_json (list): Each item must have 'audio' (base64 string) and 'transcription' (str).
    Returns:
        tuple: (temp_dir, csv_path) where temp_dir contains 'audio/' and csv_path is the CSV file.
    """
    temp_dir = tempfile.mkdtemp(prefix="life_app_")
    audio_dir = os.path.join(temp_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    rows = []
    # Write each audio file and collect CSV rows
    for idx, row in enumerate(dataset_json):
        audio_bytes = row["audio"]
        transcription = row["transcription"]
        audio_data = base64.b64decode(audio_bytes)
        audio_filename = f"audio_{idx}.wav"
        audio_path = os.path.join(audio_dir, audio_filename)
        with open(audio_path, "wb") as af:
            af.write(audio_data)
        # Add entry for CSV: just the filename, not the full path
        rows.append({"audio": audio_filename, "transcription": transcription})
    csv_path = os.path.join(temp_dir, "data.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return temp_dir, csv_path 