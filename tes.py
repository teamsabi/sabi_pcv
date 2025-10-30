import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = load_model('sabi_cnn_model.h5')
print("Model berhasil dimuat!")

# --------------------------------------------------
# Label Kelas (sesuai folder dataset)
# Pastikan urutan sesuai train_data.class_indices
# --------------------------------------------------
labels = ['Bercak Daun', 'Daun Keriting', 'Daun Sehat', 'Layu Daun']  

# --------------------------------------------------
# Path Gambar yang Mau Diprediksi
# --------------------------------------------------
img_path = "dataset/test/Daun Keriting/Daun Keriting (7).jpg"  # ganti sesuai file gambar kamu
print(f"Mengenali gambar: {img_path}")

# --------------------------------------------------
# Preprocessing Gambar
# --------------------------------------------------
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # untuk batch 1 gambar

# --------------------------------------------------
# Prediksi
# --------------------------------------------------
predictions = model.predict(img_array)
class_index = np.argmax(predictions)
confidence = np.max(predictions)

print(f"\nHasil Prediksi: {labels[class_index]}")
print(f"Tingkat Keyakinan: {confidence*100:.2f}%")