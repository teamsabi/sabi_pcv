from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --------------------------------------------------
# Load Model Sekali di Awal
# --------------------------------------------------
MODEL_PATH = "sabi_cnn_model.h5"
model = load_model(MODEL_PATH)
labels = ['Bercak Daun', 'Daun Keriting', 'Daun Sehat', 'Layu Daun']

# Folder sementara untuk menyimpan gambar upload
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return "Flask API SABI sudah aktif!"

# --------------------------------------------------
# Endpoint untuk Prediksi 4 Gambar
# --------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist("images")  # Ambil semua file upload
    if len(files) == 0:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400

    predictions = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Preprocessing gambar
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        pred = model.predict(img_array)
        class_index = np.argmax(pred)
        confidence = float(np.max(pred))

        predictions.append({
            'nama_file': filename,
            'hasil': labels[class_index],
            'keyakinan': round(confidence * 100, 2)
        })

    # Hitung hasil dominan dari 4 gambar
    hasil_akhir = {}
    for p in predictions:
        hasil_akhir[p['hasil']] = hasil_akhir.get(p['hasil'], 0) + 1

    # Ambil hasil terbanyak (mayoritas)
    penyakit_dominan = max(hasil_akhir, key=hasil_akhir.get)

    return jsonify({
        'status': 'success',
        'hasil_masing': predictions,
        'hasil_dominan': penyakit_dominan
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)