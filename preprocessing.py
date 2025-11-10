import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = './dataset'
train_dir = os.path.join(base_dir, 'train')
output_aug_dir = './dataset_augmentasi'
os.makedirs(output_aug_dir, exist_ok=True)

# Augmentasi konfigurasi
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='reflect'
)

img_size = (224, 224)
batch_size = 32

# Loop keempat kelas secara otomatis
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    class_output_dir = os.path.join(output_aug_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    class_gen = train_datagen.flow_from_directory(
        train_dir,
        classes=[class_name],
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        save_to_dir=class_output_dir,
        save_prefix=f'{class_name}_aug',
        save_format='jpg'
    )
    
    print(f"Mulai augmentasi kelas: {class_name}")
    for i in range(5):  # jumlah batch per kelas
        next(class_gen)
    print(f"Selesai augmentasi kelas: {class_name}\n")

print("Semua kelas sudah diaugmentasi!")