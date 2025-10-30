import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = load_model('sabi_cnn_model.h5')
print("Model berhasil dimuat!")

# --------------------------------------------------
# Path Dataset Test
# --------------------------------------------------
test_dir = os.path.join('.', 'dataset', 'test')

# --------------------------------------------------
# Preprocessing Data Test
# --------------------------------------------------
img_size = (150, 150)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# --------------------------------------------------
# Evaluasi Model
# --------------------------------------------------
test_loss, test_acc = model.evaluate(test_data)
print(f"\nAkurasi pada data test: {test_acc*100:.2f}%")

# --------------------------------------------------
# Prediksi dan Laporan Klasifikasi
# --------------------------------------------------
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)

print("\nLaporan Klasifikasi:")
print(classification_report(
    test_data.classes,
    y_pred,
    target_names=list(test_data.class_indices.keys())
))

print("\nConfusion Matrix:")
cm = confusion_matrix(test_data.classes, y_pred)
print(cm)

# --------------------------------------------------
# Visualisasi Confusion Matrix
# --------------------------------------------------
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.ylabel("Label Asli")
plt.xticks(np.arange(len(test_data.class_indices)), list(test_data.class_indices.keys()), rotation=45)
plt.yticks(np.arange(len(test_data.class_indices)), list(test_data.class_indices.keys()))
plt.colorbar()
plt.show()