import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ✅ use non-GUI backend for servers/VS Code

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for saving model

# === Load features and labels ===
print("📂 Loading MFCC features...")
X = np.load("data/features/mfcc_features.npy")
y = np.load("data/features/labels.npy")

# === Encode labels (e.g., Hindi → 0, Tamil → 1, etc.) ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_
print(f"✅ Languages detected: {list(classes)}")

# === Normalize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"🧩 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# === Train model ===
print("\n🚀 Training RandomForest classifier...")
clf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)

print("\n📊 Classification Report:")
report = classification_report(y_test, y_pred, target_names=classes)
print(report)

# === Confusion matrix (saved to file) ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - MFCC Baseline")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

# Save outputs
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix.png")
print("\n🖼️  Confusion matrix saved to results/confusion_matrix.png")

# === Save model and label encoder ===
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/mfcc_baseline_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("💾 Model, label encoder, and scaler saved in 'models/' folder.")
print("\n✅ Training complete! Baseline model ready.")
