import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# INFORMASI JUMLAH DATASET SEBELUM AUGMENTASI
# --------------------------------------------------
print("=== JUMLAH DATASET SEBELUM AUGMENTASI ===")
train_counts = {
    "Daun Sehat": 644,
    "Bercak Daun": 453,
    "Layu Daun": 354,
    "Daun Keriting": 719
}
test_counts = {
    "Daun Sehat": 85,
    "Bercak Daun": 40,
    "Layu Daun": 50,
    "Daun Keriting": 90
}
val_counts = {
    "Daun Sehat": 110,
    "Bercak Daun": 181,
    "Layu Daun": 90,
    "Daun Keriting": 193
}

train_total = sum(train_counts.values())
test_total = sum(test_counts.values())
val_total = sum(val_counts.values())
grand_total = train_total + test_total + val_total

print(f"Train: {train_total} data")
print(f"Valid: {val_total} data")
print(f"Test:  {test_total} data")
print(f"TOTAL: {grand_total} data\n")

# --------------------------------------------------
# ASUMSI: AUGMENTASI MENINGKATKAN JUMLAH DATA TRAINING
# (Contoh: dari 2170 → bertambah 30 kali lipat variasi gambar)
# --------------------------------------------------
augmentation_factor = 30  # kamu bisa sesuaikan dengan tingkat variasi yang digunakan
augmented_train_total = train_total + (train_total * augmentation_factor)

print("=== PERKIRAAN JUMLAH DATASET SETELAH AUGMENTASI ===")
print(f"Train (setelah augmentasi): {augmented_train_total:,} data")
print(f"Valid (tetap): {val_total} data")
print(f"Test (tetap):  {test_total} data")
print(f"TOTAL (setelah augmentasi): {augmented_train_total + val_total + test_total:,} data\n")

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
    rotation_range=35,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

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

num_classes = len(train_data.class_indices)
print(f"\nJumlah kelas: {num_classes}")
print("Kelas:", train_data.class_indices)

# --------------------------------------------------
# Arsitektur CNN (Optimized)
# --------------------------------------------------
model = Sequential([
    # Blok 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    # Blok 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Blok 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    # Blok tambahan
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.35),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(num_classes, activation='softmax')
])

# --------------------------------------------------
# Kompilasi Model
# --------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------
# Callback Penurun Learning Rate Otomatis
# --------------------------------------------------
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# --------------------------------------------------
# Training
# --------------------------------------------------
epochs = 20
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[lr_reducer]
)

# --------------------------------------------------
# Evaluasi
# --------------------------------------------------
test_loss, test_acc = model.evaluate(test_data)
print(f"\nAkurasi pada data test: {test_acc*100:.2f}%")

# --------------------------------------------------
# Visualisasi
# --------------------------------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Akurasi Training vs Validasi')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Training vs Validasi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Confusion Matrix & Report
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
print(confusion_matrix(test_data.classes, y_pred))

# --------------------------------------------------
# Simpan Model
# --------------------------------------------------
model.save('sabi_cnn_model_v4.h5')
print("\nModel CNN disimpan sebagai sabi_cnn_model_v4.h5")