import os
import cv2
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout


def load_fer2013(dataset_path):
    data, labels = [], []
    emotions = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    for emotion, label in emotions.items():
        emotion_path = os.path.join(dataset_path, emotion)
        for img_name in os.listdir(emotion_path):
            img = cv2.imread(os.path.join(emotion_path, img_name), cv2.IMREAD_GRAYSCALE)
            data.append(cv2.resize(img, (48, 48)))
            labels.append(label)
    return np.array(data), np.array(labels)



dataset_path = "dataset/train"
x, y = load_fer2013(dataset_path)
x = x.reshape(-1, 48, 48, 1) / 255.0
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)
datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
datagen.fit(x_train)
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(7, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          validation_data=(x_val, y_val),
          epochs=50,
          callbacks=[reduce_lr, early_stopping])

model.save("emotion_recognition_model.h5")