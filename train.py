import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
data_dir = 'dataset'
categories = ['broadleaf', 'grass', 'soil', 'soybean']
train_dir = 'train_data'
val_dir = 'val_data'

# Create train and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

# Split data into training and validation sets
for category in categories:
    category_path = os.path.join(data_dir, category)
    images = os.listdir(category_path)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    for image in train_images:
        shutil.copy(os.path.join(category_path, image), os.path.join(train_dir, category, image))

    for image in val_images:
        shutil.copy(os.path.join(category_path, image), os.path.join(val_dir, category, image))


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(categories), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint callback to save the model
checkpoint = ModelCheckpoint('weed_identification_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10,
    callbacks=[checkpoint]
)
