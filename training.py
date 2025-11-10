import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Path Dataset
base_dir = os.path.join('.', 'dataset')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# YOLOv8 (deteksi daun sebelum training)
print("Inisialisasi YOLOv8...")
yolo_model = YOLO('yolov8n.pt')  # model dasar YOLOv8 kecil

# Contoh deteksi 1 gambar
sample_image = os.path.join(test_dir, 'Daun Keriting', os.listdir(os.path.join(test_dir, 'Daun Keriting'))[0])
results = yolo_model.predict(sample_image, save=True, project='output', name='yolo_detection')
print("Contoh hasil deteksi daun disimpan di folder output/yolo_detection/")

# Dataset pipeline (pengganti ImageDataGenerator)
img_size = (224, 224)
batch_size = 32

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=False
)

# Simpan nama kelas sebelum .map()
class_names = train_data.class_names
num_classes = len(class_names)
print(f"\nJumlah kelas: {num_classes}")
print("Kelas:", class_names)

# Normalisasi piksel
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# Prefetch (biar lebih cepat)
train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.prefetch(buffer_size=tf.data.AUTOTUNE)

# Arsitektur CNN EfficientNetB0
print("Membangun model EfficientNetB0...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False  # freeze layer awal

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Kompilasi Model
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback Learning Rate
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# Training
epochs = 25
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[lr_reducer]
)

# Evaluasi Model
test_loss, test_acc = model.evaluate(test_data)
print(f"\nAkurasi pada data test: {test_acc*100:.2f}%")

# Visualisasi Hasil Training
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

# Prediksi dan laporan
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)

true_labels = np.concatenate([y for x, y in test_data], axis=0)
y_true = np.argmax(true_labels, axis=1)

print("\nLaporan Klasifikasi:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Simpan Model
os.makedirs('models', exist_ok=True)
model.save('models/sabi_efficientnet_yolo.h5')
print("\nModel disimpan sebagai models/sabi_efficientnet_yolo.h5")
