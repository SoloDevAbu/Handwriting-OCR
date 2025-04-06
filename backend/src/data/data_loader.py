import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
from typing import Tuple, Generator
from loguru import logger

class DataLoader:
    def __init__(self, data_dir: str, batch_size: int = 32, image_size: Tuple[int, int] = (512, 512)):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Verify data directory structure
        self._verify_data_structure()
    
    def _verify_data_structure(self):
        """Verify the expected data directory structure exists"""
        required_dirs = ['train', 'validation', 'test']
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                raise ValueError(f"Required directory {dir_path} not found")
            
            # Check for images and annotations
            images_dir = dir_path / 'images'
            labels_dir = dir_path / 'labels'
            if not (images_dir.exists() and labels_dir.exists()):
                raise ValueError(f"Missing images or labels directory in {dir_path}")
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image"""
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        return image.astype(np.float32) / 255.0
    
    def _load_annotation(self, label_path: str) -> np.ndarray:
        """Load and preprocess annotation data"""
        # Implement based on your annotation format
        # Example: Loading bounding boxes and text
        return np.load(str(label_path))
    
    def _create_dataset(self, split: str) -> tf.data.Dataset:
        """Create a TensorFlow dataset for a specific split"""
        split_dir = self.data_dir / split
        image_dir = split_dir / 'images'
        label_dir = split_dir / 'labels'
        
        image_paths = sorted(list(image_dir.glob('*.jpg')))
        label_paths = sorted(list(label_dir.glob('*.npy')))
        
        if len(image_paths) != len(label_paths):
            raise ValueError(f"Mismatch between images and labels count in {split}")
        
        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        dataset = dataset.map(lambda x, y: (
            tf.py_function(self._load_image, [x], tf.float32),
            tf.py_function(self._load_annotation, [y], tf.float32)
        ), num_parallel_calls=tf.data.AUTOTUNE)
        
        # Configure dataset
        dataset = dataset.shuffle(1000).batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_training_data(self) -> tf.data.Dataset:
        """Get the training dataset"""
        logger.info("Loading training data...")
        return self._create_dataset('train')
    
    def get_validation_data(self) -> tf.data.Dataset:
        """Get the validation dataset"""
        logger.info("Loading validation data...")
        return self._create_dataset('validation')
    
    def get_test_data(self) -> tf.data.Dataset:
        """Get the test dataset"""
        logger.info("Loading test data...")
        return self._create_dataset('test')

    @property
    def data_info(self) -> dict:
        """Get information about the dataset"""
        info = {}
        for split in ['train', 'validation', 'test']:
            split_dir = self.data_dir / split / 'images'
            info[split] = len(list(split_dir.glob('*.jpg')))
        return info