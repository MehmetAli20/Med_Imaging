import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import random
import shutil

def create_model(input_shape=(224, 224, 3)):
    # ResNet50V2 modelini temel alarak transfer learning yapıyoruz
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Model katmanlarını donduruyoruz
    for layer in base_model.layers:
        layer.trainable = False
    
    # Yeni katmanları ekliyoruz
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def prepare_subset_dataset(normal_dir, abnormal_dir, subset_dir, images_per_class=100):
    """Veri setinden alt küme oluştur"""
    # Alt küme dizinlerini oluştur
    subset_normal_dir = os.path.join(subset_dir, 'NORMAL')
    subset_abnormal_dir = os.path.join(subset_dir, 'PNEUMONIA')
    os.makedirs(subset_normal_dir, exist_ok=True)
    os.makedirs(subset_abnormal_dir, exist_ok=True)
    
    # Normal görüntüleri kopyala
    normal_images = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_normal = random.sample(normal_images, min(images_per_class, len(normal_images)))
    for img in selected_normal:
        shutil.copy2(os.path.join(normal_dir, img), os.path.join(subset_normal_dir, img))
    
    # Anormal görüntüleri kopyala
    abnormal_images = [f for f in os.listdir(abnormal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_abnormal = random.sample(abnormal_images, min(images_per_class, len(abnormal_images)))
    for img in selected_abnormal:
        shutil.copy2(os.path.join(abnormal_dir, img), os.path.join(subset_abnormal_dir, img))
    
    print(f"Normal görüntü sayısı: {len(selected_normal)}")
    print(f"Anormal görüntü sayısı: {len(selected_abnormal)}")

def train_model(normal_dir, abnormal_dir, model_save_path='models/medical_image_model.h5', images_per_class=200):
    # Alt küme dizini
    subset_dir = 'chest_xray/subset'
    
    # Alt küme veri setini hazırla
    prepare_subset_dataset(normal_dir, abnormal_dir, subset_dir, images_per_class)
    
    # Model oluştur
    model = create_model()
    
    # Model derleme
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Veri artırma ve ön işleme
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Eğitim ve doğrulama veri setlerini oluştur
    train_generator = train_datagen.flow_from_directory(
        subset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        subset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    ]
    
    # Modeli eğit
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=30,
        callbacks=callbacks
    )
    
    # Alt küme dizinini temizle
    shutil.rmtree(subset_dir)
    
    return model, history

if __name__ == "__main__":
    # Model eğitimi
    model, history = train_model(
        normal_dir='chest_xray/train/NORMAL',
        abnormal_dir='chest_xray/train/PNEUMONIA',
        images_per_class=200  # Her sınıftan 200 görüntü kullan
    )
    
    # Eğitim sonuçlarını yazdır
    print("\nEğitim tamamlandı!")
    print(f"Son eğitim doğruluğu: {history.history['accuracy'][-1]:.4f}")
    print(f"Son doğrulama doğruluğu: {history.history['val_accuracy'][-1]:.4f}") 