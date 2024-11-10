import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np
import os

# Directory path (Update to your path)
base_dir = "C:/Users/dell/Downloads/one last try/data"  # Change this to your path

# Parameters
img_size = (128, 128)
batch_size = 32
epochs = 50  # Number of epochs for training

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,            # Normalize pixel values between 0 and 1
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load images from directory structure
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

# Step 1: Compute class weights based on the distribution of the classes in the training data
class_weights = class_weight.compute_class_weight(
    'balanced',  # Option to compute class weights based on the distribution of classes
    classes=np.unique(train_generator.classes),  # Classes in the dataset
    y=train_generator.classes  # The target labels from the generator
)

# Convert the class weights into a dictionary mapping class index to class weight
class_weights = dict(zip(np.unique(train_generator.classes), class_weights))


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # Update this line to match the 7 classes in your dataset
    ])
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 2: Train the model using class weights
history = model.fit(
    train_generator,
    epochs=epochs,
    class_weight=class_weights  # Pass the class weights here
)

# Save the model
model.save('asl_model.h5')
print("Model saved as 'asl_model.h5'")

# Optionally, print the final accuracy
final_accuracy = history.history['accuracy'][-1]
print(f"Final Training Accuracy: {final_accuracy:.4f}")
