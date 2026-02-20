from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)
CORS(app)   # 🔥 allows frontend to talk to backend

# ---------------- PATHS ----------------
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'models', 'fruit_classifier.h5')
static_dir = os.path.join(base_path, 'static')

os.makedirs(static_dir, exist_ok=True)

# ---------------- LOAD MODEL ----------------
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

# ---------------- CLASS LABELS ----------------
class_names = [
    'freshapples','freshbanana','freshcapsicum','freshcucumber',
    'freshokra','freshoranges','freshpotato','freshtomato',
    'rottenapples','rottenbanana','rottencapsicum','rottencucumber',
    'rottenokra','rottenoranges','rottenpotato','rottentomato'
]

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({'error':'Model not loaded'}),500

    if 'file' not in request.files:
        return jsonify({'error':'No file uploaded'}),400

    file=request.files['file']

    if file.filename=='':
        return jsonify({'error':'Empty filename'}),400

    try:
        # safer filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(static_dir, filename)
        file.save(filepath)

        # preprocess
        img=image.load_img(filepath,target_size=(128,128))
        img_array=image.img_to_array(img)/255.0
        img_array=np.expand_dims(img_array,axis=0)

        # predict
        preds=model.predict(img_array)
        idx=int(np.argmax(preds[0]))
        confidence=float(preds[0][idx])

        label=class_names[idx]
        label=label.replace('fresh','Fresh ').replace('rotten','Rotten ')

        return jsonify({
            "predicted_class":label,
            "confidence":confidence
        })

    except Exception as e:
        print("❌ Prediction error:",e)
        return jsonify({'error':str(e)}),500


# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)