import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# Load metadata
metadata = pd.read_csv("data/metadata.csv")

# Output files
os.makedirs("data/features", exist_ok=True)

mfcc_features = []
labels = []

print("🎧 Extracting MFCC features...")

for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
    path = row["path"]
    label = row["label"]

    try:
        # Load audio
        y, sr = librosa.load(path, sr=16000)

        # Extract MFCCs (40 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=400, hop_length=160)

        # Mean + Std pooling (aggregate over time)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        feature_vector = np.concatenate((mfcc_mean, mfcc_std))

        mfcc_features.append(feature_vector)
        labels.append(label)
    except Exception as e:
        print(f"⚠️ Skipping {path}: {e}")

# Convert to numpy arrays
X = np.array(mfcc_features)
y = np.array(labels)

# Save to disk
np.save("data/features/mfcc_features.npy", X)
np.save("data/features/labels.npy", y)

print(f"\n✅ Extracted MFCC features for {len(X)} files.")
print("📁 Saved to data/features/")
