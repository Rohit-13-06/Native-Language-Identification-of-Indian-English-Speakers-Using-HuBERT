import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from tqdm import tqdm

# --- Configuration ---
MODEL_ID = "facebook/hubert-base-ls960"
DATASET_CSV = "data/metadata.csv"
OUTPUT_FEATURES = "data/features/hubert_features.npy"
OUTPUT_LABELS = "data/features/hubert_labels.npy"
os.makedirs("data/features", exist_ok=True)

# --- Device selection ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Loading HuBERT model on {device} ...")

# --- Load pretrained HuBERT model ---
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
hubert = HubertModel.from_pretrained(MODEL_ID).to(device)
hubert.eval()

# --- Load metadata ---
metadata = pd.read_csv(DATASET_CSV)
print(f"🎧 Extracting HuBERT embeddings from {len(metadata)} files...")

hubert_features = []
labels = []

# --- Extraction loop ---
for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
    path = row["path"]
    label = row["label"]

    try:
        # ✅ Safe audio loading (no torchcodec / ffmpeg issues)
        wav, sr = sf.read(path)
        if len(wav.shape) > 1:  # stereo → mono
            wav = np.mean(wav, axis=1)
        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

        # Resample to 16k if needed
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)

        # --- Feature extraction ---
        inputs = feature_extractor(
            wav.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_values.to(device)

        with torch.no_grad():
            outputs = hubert(inputs)
            # Mean-pool last hidden layer (768-dim embedding)
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

        hubert_features.append(emb)
        labels.append(label)

    except Exception as e:
        print(f"⚠️ Skipping {path}: {e}")

# --- Save embeddings ---
X = np.array(hubert_features)
y = np.array(labels)
np.save(OUTPUT_FEATURES, X)
np.save(OUTPUT_LABELS, y)

print(f"\n✅ Extracted and saved {len(X)} HuBERT embeddings.")
print(f"📁 Feature file: {OUTPUT_FEATURES}")
print(f"🧾 Labels file: {OUTPUT_LABELS}")
print(f"Feature shape: {X.shape}")
