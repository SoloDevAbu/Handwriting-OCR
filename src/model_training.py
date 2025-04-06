import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_model(input_shape, num_classes):
    """
    Create a CNN model for image classification
    """
    model = models.Sequential([
        # First convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(X, y, model_path, epochs=10, batch_size=32):
    """
    Train the model on the given dataset
    """
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile the model
    model = create_model(input_shape=(28, 28, 1), num_classes=len(np.unique(y)))
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(X_val, y_val))
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    
    return model, history

def main():
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    models_dir = base_dir / 'models'
    
    # Load processed data
    print("Loading processed data...")
    X_handwritten = np.load(processed_dir / 'handwritten' / 'X.npy')
    y_handwritten = np.load(processed_dir / 'handwritten' / 'y.npy')
    
    X_keys = np.load(processed_dir / 'keys' / 'X.npy')
    y_keys = np.load(processed_dir / 'keys' / 'y.npy')
    
    # Reshape data for CNN (add channel dimension)
    X_handwritten = X_handwritten.reshape(-1, 28, 28, 1)
    X_keys = X_keys.reshape(-1, 28, 28, 1)
    
    # Train handwritten text model
    print("Training handwritten text model...")
    handwritten_model_path = models_dir / 'handwritten_model'
    handwritten_model, handwritten_history = train_model(
        X_handwritten, y_handwritten,
        handwritten_model_path,
        epochs=10
    )
    
    # Train key model
    print("Training key model...")
    key_model_path = models_dir / 'key_model'
    key_model, key_history = train_model(
        X_keys, y_keys,
        key_model_path,
        epochs=10
    )
    
    print("Training completed! Models saved in the 'models' directory.")

if __name__ == "__main__":
    main() 