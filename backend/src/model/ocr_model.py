import tensorflow as tf
import tensorflow_hub as hub
import pytesseract
from loguru import logger
import numpy as np
from typing import Optional

class OCRModel:
    def __init__(self):
        """Initialize the OCR model with both TensorFlow and Tesseract backends"""
        logger.info("Initializing OCR model...")
        self.tf_model: Optional[tf.keras.Model] = None
        self._initialize_tf_model()
        self._configure_tesseract()
        
    def _initialize_tf_model(self):
        """Initialize and load the TensorFlow model for text detection"""
        try:
            # Load a pre-trained text detection model from TF Hub
            detector_url = "https://tfhub.dev/tensorflow/craft/1"
            self.tf_model = hub.load(detector_url)
            logger.info("TensorFlow model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            raise
    
    def _configure_tesseract(self):
        """Configure Tesseract OCR settings"""
        try:
            # Configure Tesseract parameters for better accuracy
            custom_config = r'--oem 3 --psm 6'
            pytesseract.get_tesseract_version()
            logger.info("Tesseract configured successfully")
        except Exception as e:
            logger.error(f"Error configuring Tesseract: {e}")
            raise
    
    def detect_text_regions(self, image: np.ndarray) -> list:
        """
        Detect text regions in the image using the TensorFlow model.
        
        Args:
            image: Preprocessed image as numpy array
        Returns:
            List of detected text regions
        """
        # Prepare image for the model
        img = tf.cast(image, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)
        
        # Get text detection predictions
        detection_map = self.tf_model(img)
        return self._process_detection_map(detection_map[0])
    
    def _process_detection_map(self, detection_map: tf.Tensor) -> list:
        """Process the detection map to get text region coordinates"""
        # Convert detection map to numpy and threshold
        detection_map = detection_map.numpy()
        thresh = 0.2
        mask = detection_map > thresh
        
        # Find contours in the mask
        regions = []
        for i in range(mask.shape[-1]):
            layer = mask[:, :, i].astype(np.uint8)
            contours, _ = cv2.findContours(
                layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 10 and h > 10:  # Filter out small regions
                    regions.append((x, y, x + w, y + h))
        
        return regions
    
    def recognize_text(self, image: np.ndarray, regions: list = None) -> str:
        """
        Perform OCR on the image using Tesseract.
        
        Args:
            image: Preprocessed image
            regions: Optional list of text regions to process
        Returns:
            Recognized text string
        """
        try:
            if regions:
                # Process each region separately
                text_parts = []
                for (x1, y1, x2, y2) in regions:
                    roi = image[y1:y2, x1:x2]
                    text = pytesseract.image_to_string(roi)
                    text_parts.append(text.strip())
                return ' '.join(text_parts)
            else:
                # Process the entire image
                return pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Error in text recognition: {e}")
            raise