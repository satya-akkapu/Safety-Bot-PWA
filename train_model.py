import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ================= CONFIG =================
SR = 22050
DURATION = 3
SAMPLES = SR * DURATION
N_MELS = 128

DEV_AUDIO_DIR  = r"C:\Users\DELL\Downloads\emergency_bot\datasets\FSD50K.dev_audio"
EVAL_AUDIO_DIR = r"C:\Users\DELL\Downloads\emergency_bot\datasets\FSD50K.eval_audio"

DEV_META  = r"C:\Users\DELL\Downloads\emergency_bot\datasets\FSD50K.ground_truth\balanced_dev.csv"
EVAL_META = r"C:\Users\DELL\Downloads\emergency_bot\datasets\FSD50K.ground_truth\eval.csv"

MODEL_PATH = r"C:\Users\DELL\Downloads\emergency_bot\models\emergency_model.h5"

EMERGENCY_CLASSES = [
    "Siren",
    "Fire_alarm",
    "Smoke_alarm",
    "Explosion",
    "Glass",
    "Screaming",
    "Alarm"
]

# ================= FEATURE EXTRACTION =================
def extract_features(path):
    y, _ = librosa.load(path, sr=SR, mono=True)

    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel = librosa.power_to_db(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    return mel[..., np.newaxis]

# ================= LOAD CSV + AUDIO =================
def load_dataset(csv_path, audio_dir):
    meta = pd.read_csv(csv_path)
    X, y = [], []

    for _, row in meta.iterrows():
        fname = str(row["fname"])
        if not fname.endswith(".wav"):
            fname += ".wav"

        audio_path = os.path.join(audio_dir, fname)
        if not os.path.exists(audio_path):
            continue

        labels = row["labels"]
        is_emergency = any(cls in labels for cls in EMERGENCY_CLASSES)

        try:
            X.append(extract_features(audio_path))
            y.append(1 if is_emergency else 0)
        except:
            continue

    return X, y

# ================= LOAD ALL DATA =================
print("🔄 Loading DEV dataset...")
X_dev, y_dev = load_dataset(DEV_META, DEV_AUDIO_DIR)

print("🔄 Loading EVAL dataset...")
X_eval, y_eval = load_dataset(EVAL_META, EVAL_AUDIO_DIR)

X = np.array(X_dev + X_eval)
y = np.array(y_dev + y_eval)

print("✅ Total samples:", len(X))
print("🚨 Emergency:", np.sum(y))
print("🔕 Non-Emergency:", len(y) - np.sum(y))

# ================= TRAIN / VAL SPLIT =================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================= MODEL =================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=X.shape[1:]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN =================
print("🚀 Training model...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32
)

# ================= SAVE =================
model.save(MODEL_PATH)
print("✅ Model saved at:", MODEL_PATH)
