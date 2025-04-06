import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_history(history, title):
    """
    Plot training and validation accuracy/loss
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_directory_structure():
    """
    Create the necessary directory structure for the project
    """
    base_dir = Path(__file__).parent.parent
    directories = [
        'data/raw/handwritten',
        'data/raw/keys',
        'data/processed/handwritten',
        'data/processed/keys',
        'models',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(base_dir / directory, exist_ok=True)
    
    print("Directory structure created successfully!")

def save_model_metrics(model, history, model_name):
    """
    Save model metrics to a text file
    """
    metrics_dir = Path(__file__).parent.parent / 'models' / 'metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = metrics_dir / f'{model_name}_metrics.txt'
    
    with open(metrics_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
    
    print(f"Metrics saved to {metrics_file}") 