import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # so matplotlib doesn't need a GUI
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Load HuBERT features ===
print("📂 Loading HuBERT embeddings...")
X = np.load("data/features/hubert_features.npy")
y = np.load("data/features/hubert_labels.npy")

# === Encode labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_
print(f"✅ Languages detected: {list(classes)}")

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"🧩 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# === Train RandomForest classifier ===
print("🚀 Training RandomForest classifier on HuBERT embeddings...")
clf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# === Evaluate model ===
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

# === Confusion matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - HuBERT Baseline")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix_hubert.png")

# === Save model ===
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/hubert_baseline_model.pkl")
joblib.dump(label_encoder, "models/hubert_label_encoder.pkl")
joblib.dump(scaler, "models/hubert_scaler.pkl")

print("\n✅ HuBERT baseline training complete!")
print("🖼️ Confusion matrix saved to results/confusion_matrix_hubert.png")
print("💾 Model and encoders saved in models/")
