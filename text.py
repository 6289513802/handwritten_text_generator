import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import matplotlib.pyplot as plt

# Load and preprocess the DeepWriting dataset
def load_data(data_dir):
    images = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):  # Assuming images are in PNG format
            img_path = os.path.join(data_dir, filename)
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 64))  # Resize to a fixed size
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
            
            # Extract label from filename (assuming filename format is 'label.png')
            label = filename.split('.')[0]
            labels.append(label)
    
    return np.array(images), labels

# Convert labels to sequences of integers
def labels_to_sequences(labels, char_to_int):
    sequences = []
    for label in labels:
        seq = [char_to_int[char] for char in label]
        sequences.append(seq)
    return sequences

# Create a mapping from characters to integers
def create_char_mapping(labels):
    chars = sorted(set(''.join(labels)))
    char_to_int = {char: i + 1 for i, char in enumerate(chars)}  # Start from 1
    int_to_char = {i + 1: char for i, char in enumerate(chars)}
    return char_to_int, int_to_char

# Define the model architecture
def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Reshape((input_shape[0], input_shape[1], 1)))  # Reshape for CNN
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer for character probabilities
    return model

# Train the model
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Generate handwritten text
def generate_handwriting(model, input_text, char_to_int, int_to_char):
    input_seq = [char_to_int[char] for char in input_text]
    input_seq = np.array(input_seq).reshape(1, -1)  # Reshape for model input
    predictions = model.predict(input_seq)
    predicted_indices = np.argmax(predictions, axis=1)
    generated_text = ''.join([int_to_char[i] for i in predicted_indices])
    return generated_text

# Main execution
if __name__ == "__main__":
    data_dir = 'path/to/deepwriting/dataset'  # Update with your dataset path
    images, labels = load_data(data_dir)
    
    # Create character mapping
    char_to_int, int_to_char = create_char_mapping(labels)
    
    # Convert labels to sequences
    sequences = labels_to_sequences(labels, char_to_int)
    
    # Prepare data for training
    X_train = images.reshape(images.shape[0], 64, 128, 1)  # Reshape for CNN input
    y_train = np.array(sequences)  # Convert to numpy array
    
    # Create and train the model
    model = create_model((64, 128, 1), len(char_to_int) + 1)  # +1 for padding
    train_model(model, X_train, y_train)
    
    # Generate handwritten text
    input_text = "Hello"
    generated_text = generate_handwriting(model, input_text, char_to_int, int_to_char)
    print("Generated Handwritten Text", generated_text)
