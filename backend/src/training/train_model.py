import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import json
from loguru import logger

class ModelTrainer:
    def __init__(self, data_dir, output_dir, image_size=(28, 28), batch_size=32):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Directory containing the training data
            output_dir: Directory to save the trained model and results
            image_size: Size to resize images to
            batch_size: Batch size for training
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "models", exist_ok=True)
        os.makedirs(self.output_dir / "metrics", exist_ok=True)
        os.makedirs(self.output_dir / "plots", exist_ok=True)

    def prepare_data(self, validation_split=0.2, test_split=0.1):
        """
        Prepare the dataset by preprocessing images and splitting into train/validation/test sets.
        
        Args:
            validation_split: Proportion of data to use for validation
            test_split: Proportion of data to use for testing
        
        Returns:
            Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test, and class_names
        """
        logger.info(f"Processing dataset from {self.data_dir}")
        
        # Process all images and gather labels
        processed_images = []
        labels = []
        label_map = {}
        label_count = 0
        
        # Get all subdirectories (each represents a class)
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        for subdir in subdirs:
            label_name = subdir.name
            if label_name not in label_map:
                label_map[label_name] = label_count
                label_count += 1
            
            # Process all images in this class directory
            image_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")) + list(subdir.glob("*.jpeg"))
            
            logger.info(f"Found {len(image_files)} images in class {label_name}")
            
            for img_path in image_files:
                try:
                    # Preprocess image
                    img = self._preprocess_image(str(img_path))
                    processed_images.append(img)
                    labels.append(label_map[label_name])
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {str(e)}")
        
        # Convert to numpy arrays
        X = np.array(processed_images)
        y = np.array(labels)
        
        # Create reverse mapping from label index to name
        class_names = {v: k for k, v in label_map.items()}
        
        # Split the data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_split, stratify=y, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=validation_split/(1-test_split),
            stratify=y_train_val, 
            random_state=42
        )
        
        # Reshape data for CNN (add channel dimension)
        X_train = X_train.reshape(-1, self.image_size[0], self.image_size[1], 1)
        X_val = X_val.reshape(-1, self.image_size[0], self.image_size[1], 1)
        X_test = X_test.reshape(-1, self.image_size[0], self.image_size[1], 1)
        
        # Save class mapping
        with open(self.output_dir / "class_mapping.json", "w") as f:
            json.dump(class_names, f, indent=4)
        
        logger.info(f"Data preparation complete. Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "class_names": class_names
        }
    
    def _preprocess_image(self, image_path):
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        resized = cv2.resize(gray, self.image_size)
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        return normalized
    
    def create_model(self, input_shape, num_classes):
        """
        Create a CNN model for image classification.
        
        Args:
            input_shape: Shape of input images
            num_classes: Number of classes to predict
        
        Returns:
            Compiled model
        """
        model = models.Sequential([
            # First convolutional layer
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional layer
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional layer
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Dropout to prevent overfitting
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data, epochs=15):
        """
        Train the model.
        
        Args:
            data: Dictionary containing the prepared data
            epochs: Number of epochs to train for
        
        Returns:
            Trained model and training history
        """
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        class_names = data["class_names"]
        
        # Create model
        input_shape = (self.image_size[0], self.image_size[1], 1)
        num_classes = len(class_names)
        model = self.create_model(input_shape, num_classes)
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / "models" / "best_model"),
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.00001
            )
        ]
        
        # Train model
        logger.info(f"Starting model training for {epochs} epochs")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        # Save final model
        model.save(str(self.output_dir / "models" / "final_model"))
        
        # Save training history
        self._save_history(history)
        
        # Evaluate on test set
        self._evaluate_model(model, data)
        
        return model, history
    
    def _save_history(self, history):
        """
        Save and plot training history.
        
        Args:
            history: Training history
        """
        # Save history to file
        history_dict = {
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']],
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']]
        }
        
        with open(self.output_dir / "metrics" / "training_history.json", "w") as f:
            json.dump(history_dict, f, indent=4)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(str(self.output_dir / "plots" / "training_history.png"))
    
    def _evaluate_model(self, model, data):
        """
        Evaluate the model on the test set.
        
        Args:
            model: Trained model
            data: Dictionary containing the prepared data
        """
        X_test, y_test = data["X_test"], data["y_test"]
        class_names = data["class_names"]
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test)
        
        # Get predictions
        predictions = model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred)
        
        # Save evaluation results
        with open(self.output_dir / "metrics" / "evaluation_results.txt", "w") as f:
            f.write(f"Test accuracy: {test_acc:.4f}\n")
            f.write(f"Test loss: {test_loss:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred, 
                                         target_names=[class_names[i] for i in range(len(class_names))]))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = [class_names[i] for i in range(len(class_names))]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        plt.savefig(str(self.output_dir / "plots" / "confusion_matrix.png"))
        
        logger.info(f"Model evaluation complete. Test accuracy: {test_acc:.4f}")

def main():
    """
    Main function to run model training.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a handwritten text recognition model")
    parser.add_argument("--data-dir", required=True, help="Directory containing training data")
    parser.add_argument("--output-dir", default="training_output", help="Directory to save results")
    parser.add_argument("--image-size", default="28,28", help="Size to resize images to (width,height)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train for")
    
    args = parser.parse_args()
    
    # Parse image size
    image_size = tuple(map(int, args.image_size.split(",")))
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=image_size,
        batch_size=args.batch_size
    )
    
    # Prepare data
    data = trainer.prepare_data()
    
    # Train model
    model, history = trainer.train(data, epochs=args.epochs)
    
    logger.info(f"Training complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 