import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import requests
from datetime import datetime

# --------------------------------------------------
# Load model CNN
# --------------------------------------------------
model = load_model('sabi_cnn_model.h5')
labels = ['Bercak Daun', 'Daun Keriting', 'Daun Sehat', 'Layu Daun']

# --------------------------------------------------
# Endpoint API Laravel
# --------------------------------------------------
API_URL = "http://localhost:8000/api/deteksi"  # Ganti dengan URL API Laravel kamu

# --------------------------------------------------
# Inisialisasi webcam
# --------------------------------------------------
camera = cv2.VideoCapture(0)  # 0 = default webcam

if not camera.isOpened():
    print("Webcam tidak ditemukan!")
    exit()

print("Sistem monitoring otomatis dimulai...\n")

# --------------------------------------------------
# Fungsi ambil gambar dan klasifikasi
# --------------------------------------------------
def capture_and_predict():
    ret, frame = camera.read()
    if not ret:
        print("Gagal menangkap gambar.")
        return

    # Simpan gambar dengan timestamp
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame) 

    # Preprocessing untuk CNN
    img = cv2.resize(frame, (150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    label = labels[class_index]
    confidence = np.max(pred) * 100

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {label} ({confidence:.2f}%)")

    # Kirim hasil ke API Laravel
    try:
        response = requests.post(API_URL, json={
            "nama_file": filename,
            "hasil_deteksi": label,
            "akurasi": round(confidence, 2),
            "waktu": datetime.now().isoformat()
        })
        if response.status_code == 200:
            print("Hasil berhasil dikirim ke server.\n")
        else:
            print(f"Gagal kirim hasil ke server. Status: {response.status_code}")
    except Exception as e:
        print("Gagal mengirim ke server:", e)

# --------------------------------------------------
# Loop tiap 1 jam (3600 detik)
# --------------------------------------------------
try:
    while True:
        capture_and_predict()
        time.sleep(3600)  # jeda 1 jam
except KeyboardInterrupt:
    print("Monitoring dihentikan secara manual.")
    camera.release()
    cv2.destroyAllWindows()