import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
DATASET_PATH = "data/IndicAccentDB"   # main dataset folder
OUTPUT_PATH = "data/wav16k"           # resampled output folder
SAMPLE_RATE = 16000                   # required for HuBERT
os.makedirs(OUTPUT_PATH, exist_ok=True)

records = []

print("🔍 Scanning dataset folders for .wav files...\n")

# Recursively search every folder
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(".wav"):
            src = os.path.join(root, file)
            # Label = name of parent folder (e.g., Hindi, Tamil, etc.)
            label = os.path.basename(os.path.dirname(src))
            dst = os.path.join(OUTPUT_PATH, f"{label}_{file}")

            try:
                y, sr = librosa.load(src, sr=SAMPLE_RATE)
                sf.write(dst, y, SAMPLE_RATE)
                records.append((dst, label))
            except Exception as e:
                print(f"⚠️ Skipping {src}: {e}")

# Convert to DataFrame and save metadata
df = pd.DataFrame(records, columns=["path", "label"])
df.to_csv("data/metadata.csv", index=False)

print(f"\n✅ Done! Resampled {len(df)} audio files.")
print("📁 Output folder:", OUTPUT_PATH)
print("🧾 Metadata saved to: data/metadata.csv")
