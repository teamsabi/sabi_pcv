import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Path Dataset
# --------------------------------------------------
base_dir = os.path.join('.', 'dataset')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# --------------------------------------------------
# Data Preprocessing (Normalisasi + Augmentasi)
# --------------------------------------------------
img_size = (150, 150)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# --------------------------------------------------
# Info Kelas dan Jumlah Data
# --------------------------------------------------
num_classes = len(train_data.class_indices)
print(f"\nJumlah kelas terdeteksi: {num_classes}")
print("Kelas yang terdeteksi:", train_data.class_indices)

train_counts_init = {'Daun Sehat': 644, 'Bercak Daun': 453, 'Layu Daun': 354, 'Daun Keriting': 719}
val_counts_init   = {'Daun Sehat': 110, 'Bercak Daun': 181, 'Layu Daun': 90, 'Daun Keriting': 193}
test_counts_init  = {'Daun Sehat': 85, 'Bercak Daun': 40, 'Layu Daun': 50, 'Daun Keriting': 90}

print("\nJumlah Data Asli (Sebelum Augmentasi):")
print(f"Training   : {sum(train_counts_init.values())} gambar")
print(f"Validation : {sum(val_counts_init.values())} gambar")
print(f"Testing    : {sum(test_counts_init.values())} gambar")
print(f"TOTAL DATA ASLI : {sum(train_counts_init.values()) + sum(val_counts_init.values()) + sum(test_counts_init.values())}")

augmented_per_epoch = train_data.n * (1 + train_datagen.zoom_range[1] / 4 + train_datagen.rotation_range / 100)
print(f"\nPerkiraan jumlah data hasil augmentasi per epoch ≈ {int(augmented_per_epoch)} variasi gambar")

# --------------------------------------------------
# Arsitektur CNN (Kompleks + Stabil)
# --------------------------------------------------
model = Sequential([
    # Blok 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Blok 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    # Blok 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

# --------------------------------------------------
# Kompilasi Model
# --------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # sedikit lebih tinggi agar cepat konvergen
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------
# Training Model
# --------------------------------------------------
epochs = 20
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# --------------------------------------------------
# Evaluasi Model
# --------------------------------------------------
test_loss, test_acc = model.evaluate(test_data)
print(f"\n Akurasi pada data test: {test_acc*100:.2f}%")

# --------------------------------------------------
# Jumlah Data Setelah Preprocessing
# --------------------------------------------------
print("\nJumlah Data Setelah Preprocessing / Batch (Efektif Digunakan):")
print(f"Training   : {train_data.n} gambar")
print(f"Validation : {val_data.n} gambar")
print(f"Testing    : {test_data.n} gambar")
print(f"TOTAL DATA EFEKTIF : {train_data.n + val_data.n + test_data.n} gambar")

# --------------------------------------------------
# Visualisasi Training
# --------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Akurasi Training vs Validasi')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Loss Training vs Validasi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Confusion Matrix & Laporan Klasifikasi
# --------------------------------------------------
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)

print("\n Laporan Klasifikasi:")
print(classification_report(
    test_data.classes,
    y_pred,
    target_names=list(test_data.class_indices.keys())
))

print("\n Confusion Matrix:")
print(confusion_matrix(test_data.classes, y_pred))

# --------------------------------------------------
# Simpan Model
# --------------------------------------------------
model.save('sabi_cnn_model_v3.h5')
print("\n Model CNN berhasil disimpan sebagai sabi_cnn_model_v3.h5")
