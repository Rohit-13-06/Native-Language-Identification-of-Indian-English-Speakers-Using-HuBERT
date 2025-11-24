import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import joblib

# -----------------------------
#  Load saved model components
# -----------------------------
MODEL_PATH = "models/hubert_baseline_model.pkl"
ENCODER_PATH = "models/hubert_label_encoder.pkl"
SCALER_PATH = "models/hubert_scaler.pkl"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# HuBERT feature extractor
MODEL_ID = "facebook/hubert-base-ls960"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
hubert = HubertModel.from_pretrained(MODEL_ID)
hubert.eval()

# Accent → Cuisine mapping
accent_to_food = {
    "andhra_pradesh": ["Gongura Pachadi", "Pesarattu", "Pulihora"],
    "kerala": ["Appam", "Puttu", "Avial"],
    "tamil": ["Idli", "Dosa", "Pongal"],
    "karnataka": ["Bisi Bele Bath", "Ragi Mudde"],
    "gujrat": ["Dhokla", "Thepla"],
    "jharkhand": ["Litti Chokha", "Thekua"],
}

# -----------------------------
#  Streamlit UI
# -----------------------------
st.set_page_config(page_title="Accent-Aware Cuisine Recommender", page_icon="🍛")
st.title("🍛 Accent-Aware Cuisine Recommendation System")
st.markdown("Upload an English audio clip, and we'll detect your accent to recommend regional dishes!")

audio_file = st.file_uploader("🎤 Upload your audio (.wav format only):", type=["wav"])

if audio_file is not None:
    # Save the uploaded file temporarily
    wav, sr = sf.read(audio_file)
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)  # stereo → mono
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    # Extract HuBERT features
    with torch.no_grad():
        inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        outputs = hubert(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten().reshape(1, -1)

    # Scale and predict
    emb_scaled = scaler.transform(emb)
    pred = model.predict(emb_scaled)
    accent = label_encoder.inverse_transform(pred)[0]

    st.success(f"🎯 Detected Accent: **{accent.replace('_', ' ').title()}**")

    # Cuisine recommendation
    if accent in accent_to_food:
        st.markdown("### 🍽️ Recommended Regional Cuisines:")
        for dish in accent_to_food[accent]:
            st.write(f"• {dish}")
    else:
        st.info("No cuisine data available for this accent yet.")
