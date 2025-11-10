import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# --------------------------------------------------
# Path Dataset
# --------------------------------------------------
base_dir = os.path.join('.', 'dataset')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# --------------------------------------------------
# Inisialisasi YOLOv8 (untuk deteksi daun)
# --------------------------------------------------
yolo_model = YOLO('yolov8n.pt')

# --------------------------------------------------
# Preprocessing Data
# --------------------------------------------------
img_size = (224, 224)
batch_size = 32

def ensure_rgb(x):
    if x.shape[-1] == 1:
        x = tf.image.grayscale_to_rgb(x)
    return x

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=35,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    preprocessing_function=ensure_rgb
)

val_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=ensure_rgb)
test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=ensure_rgb)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', color_mode='rgb'
)
val_data = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', color_mode='rgb'
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', color_mode='rgb', shuffle=False
)

num_classes = len(train_data.class_indices)
print(f"\nJumlah kelas: {num_classes}")
print("Kelas:", train_data.class_indices)

# --------------------------------------------------
# Arsitektur EfficientNetB0 tanpa pretrained
# --------------------------------------------------
base_model = EfficientNetB0(
    weights=None,  # 🚀 Ganti jadi None agar tidak error shape mismatch
    include_top=False,
    input_shape=(224, 224, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

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
    monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6
)

# --------------------------------------------------
# Training
# --------------------------------------------------
epochs = 25
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
# Visualisasi Training vs Validasi
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
# Evaluasi Detail: Confusion Matrix & Report
# --------------------------------------------------
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)

print("\nLaporan Klasifikasi:")
print(classification_report(
    test_data.classes, y_pred,
    target_names=list(test_data.class_indices.keys())
))

print("\nConfusion Matrix:")
print(confusion_matrix(test_data.classes, y_pred))

# --------------------------------------------------
# Simpan Model
# --------------------------------------------------
model.save('sabi_efficientnet_yolo_fixed.h5')
print("\n✅ Model disimpan sebagai sabi_efficientnet_yolo_fixed.h5")