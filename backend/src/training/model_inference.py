import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import json
from loguru import logger

class CustomModelInference:
    def __init__(self, model_path):
        """
        Initialize the model for inference.
        
        Args:
            model_path: Path to the saved model
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.class_mapping = self._load_class_mapping()
        logger.info(f"Loaded model from {model_path} with {len(self.class_mapping)} classes")
    
    def _load_model(self):
        """Load the TensorFlow model."""
        try:
            model = tf.keras.models.load_model(str(self.model_path))
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError(f"Failed to load model from {self.model_path}")
    
    def _load_class_mapping(self):
        """Load the class mapping JSON file."""
        try:
            mapping_file = self.model_path.parent.parent.parent / "class_mapping.json"
            if not mapping_file.exists():
                logger.warning(f"Class mapping file not found at {mapping_file}")
                # Try to infer from model output layer
                output_layer = self.model.layers[-1]
                num_classes = output_layer.output_shape[-1]
                return {str(i): str(i) for i in range(num_classes)}
            
            with open(mapping_file, "r") as f:
                mapping = json.load(f)
            return mapping
        except Exception as e:
            logger.error(f"Error loading class mapping: {e}")
            return {}
    
    def preprocess_image(self, image_data, target_size=(28, 28)):
        """
        Preprocess an image for model input.
        
        Args:
            image_data: Raw image bytes or numpy array
            target_size: Size to resize the image to
        
        Returns:
            Preprocessed image as numpy array
        """
        # Convert bytes to numpy array if needed
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image_data
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        resized = cv2.resize(gray, target_size)
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Reshape for model input (add batch and channel dimensions)
        model_input = normalized.reshape(1, target_size[0], target_size[1], 1)
        
        return model_input
    
    def predict(self, image_data):
        """
        Run inference on an image.
        
        Args:
            image_data: Raw image bytes or numpy array
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Get model input shape from the first layer
            input_shape = self.model.layers[0].input_shape[0]
            if len(input_shape) == 4:
                target_size = (input_shape[1], input_shape[2])
            else:
                target_size = (28, 28)  # Default size
            
            # Preprocess image
            model_input = self.preprocess_image(image_data, target_size)
            
            # Run inference
            predictions = self.model.predict(model_input)
            
            # Get the class with highest probability
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            # Get class label from mapping
            class_label = self.class_mapping.get(str(class_idx), f"Class {class_idx}")
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            top_predictions = [
                {
                    "class": self.class_mapping.get(str(idx), f"Class {idx}"),
                    "confidence": float(predictions[0][idx])
                }
                for idx in top_indices
            ]
            
            return {
                "class": class_label,
                "confidence": confidence,
                "top_predictions": top_predictions
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")

def get_available_models():
    """Get list of available trained models."""
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / "training"
    
    available_models = []
    if not models_dir.exists():
        return available_models
    
    for job_dir in models_dir.iterdir():
        if not job_dir.is_dir():
            continue
        
        model_dir = job_dir / "output" / "models" / "final_model"
        if model_dir.exists():
            # Try to get class mapping
            class_mapping = {}
            mapping_file = job_dir / "output" / "class_mapping.json"
            if mapping_file.exists():
                try:
                    with open(mapping_file, "r") as f:
                        class_mapping = json.load(f)
                except:
                    pass
            
            available_models.append({
                "id": job_dir.name,
                "path": str(model_dir),
                "classes": len(class_mapping),
                "class_names": list(class_mapping.values())
            })
    
    return available_models

def load_model(model_id_or_path):
    """Load a model by ID or path."""
    if not model_id_or_path:
        raise ValueError("Model ID or path must be provided")
    
    model_path = model_id_or_path
    
    # If it's an ID, try to find the model path
    if not Path(model_id_or_path).exists():
        base_dir = Path(__file__).parent.parent.parent
        possible_model_path = base_dir / "training" / model_id_or_path / "output" / "models" / "final_model"
        
        if possible_model_path.exists():
            model_path = str(possible_model_path)
        else:
            raise ValueError(f"Model not found with ID: {model_id_or_path}")
    
    return CustomModelInference(model_path) 