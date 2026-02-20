from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# -------- PATHS (RENDER SAFE) --------
BASE_DIR = os.getcwd()   # important for Render

MODEL_PATH = os.path.join(BASE_DIR, "models", "fruit_classifier.h5")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)

print("📂 BASE DIR:", BASE_DIR)
print("📂 MODEL PATH:", MODEL_PATH)
print("📂 MODEL EXISTS:", os.path.exists(MODEL_PATH))

# -------- LOAD MODEL --------
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully.")
    else:
        print("❌ Model file NOT FOUND at:", MODEL_PATH)
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# -------- CLASS LABELS --------
class_names = [
    'freshapples','freshbanana','freshcapsicum','freshcucumber',
    'freshokra','freshoranges','freshpotato','freshtomato',
    'rottenapples','rottenbanana','rottencapsicum','rottencucumber',
    'rottenokra','rottenoranges','rottenpotato','rottentomato'
]

# -------- ROUTES --------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(STATIC_DIR, filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(128,128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])

        raw_label = class_names[idx]

        if raw_label.startswith("fresh"):
            status = "Fresh"
            fruit = raw_label.replace("fresh","").capitalize()
        else:
            status = "Rotten"
            fruit = raw_label.replace("rotten","").capitalize()

        display_label = f"{status} {fruit}"

        return jsonify({
            "predicted_class": display_label,
            "confidence": confidence,
            "status": status
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)}), 500


# -------- RUN --------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
