from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
import librosa
import tensorflow as tf
import os
import uuid

# ✅ Import SMS function
from sms import send_sms

# ================= APP SETUP =================
app = Flask(__name__, template_folder="templates", static_folder="static")

# ✅ Secure secret key from environment
app.secret_key = os.environ.get("SECRET_KEY", "fallback_secret_key")

DB_PATH = "users.db"

# ================= LOAD ML MODEL =================
MODEL_PATH = "emergency_model.h5"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Emergency CNN model loaded")
else:
    print("❌ Model file not found!")
    model = None

# ================= GLOBAL STATES =================
latest_location = {"lat": None, "lon": None}

listener_state = {
    "running": False,
    "emergency": False,
    "hits": 0,
    "probability": 0.0
}

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            last_name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            relative_phone1 TEXT,
            relative_phone2 TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= BASIC PAGES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# ================= REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.form
        password_hash = generate_password_hash(data["password"])

        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO users
                (first_name, last_name, email, password_hash,
                 relative_phone1, relative_phone2, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data["first_name"],
                data["last_name"],
                data["email"],
                password_hash,
                data.get("relative_phone1"),
                data.get("relative_phone2"),
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
            return redirect(url_for("home"))

        except sqlite3.IntegrityError:
            flash("Email already registered")

    return render_template("register.html")

# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    return redirect(url_for("login"))

# ================= LOCATION =================
@app.route("/save_location", methods=["POST"])
def save_location():
    data = request.json
    latest_location["lat"] = data.get("lat")
    latest_location["lon"] = data.get("lon")
    return jsonify({"status": "location saved"})

@app.route("/get_location")
def get_location():
    return jsonify(latest_location)

# ================= AUDIO → ML PREDICTION =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded", "probability": 0})

        if "audio" not in request.files:
            return jsonify({"probability": 0})

        audio_file = request.files["audio"]

        filename = f"input_{uuid.uuid4()}.wav"
        audio_file.save(filename)

        y, sr = librosa.load(filename, sr=16000, mono=True)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db[:128, :128]

        if mel_db.shape[1] < 128:
            mel_db = np.pad(
                mel_db,
                ((0, 0), (0, 128 - mel_db.shape[1])),
                mode="constant"
            )

        # ✅ Safe normalization
        min_val = mel_db.min()
        max_val = mel_db.max()

        if max_val - min_val != 0:
            mel_db = (mel_db - min_val) / (max_val - min_val)
        else:
            mel_db = np.zeros_like(mel_db)

        X = mel_db.reshape(1, 128, 128, 1)

        prediction = model.predict(X)[0][0]
        probability = float(prediction)

        is_emergency = probability >= 0.70

        if is_emergency:
            lat = latest_location.get("lat")
            lon = latest_location.get("lon")

            if lat and lon:
                location_link = f"https://maps.google.com/?q={lat},{lon}"
            else:
                location_link = "Location not available"

            send_sms(
                to_number="+917702312244",
                name="User",
                location=location_link
            )

        os.remove(filename)

        return jsonify({
            "probability": round(probability, 3),
            "emergency": is_emergency
        })

    except Exception as e:
        print("🔥 PREDICT ERROR:", e)
        return jsonify({"error": str(e), "probability": 0})

# ================= STATUS =================
@app.route("/listener_status")
def listener_status():
    return jsonify(listener_state)

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)