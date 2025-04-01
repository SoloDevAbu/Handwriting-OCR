import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'backend' / 'src'))

from preprocessing.image_processor import preprocess_image, compute_skew, rotate_image

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a simple test image
        self.test_image = np.ones((100, 100), dtype=np.uint8) * 255
        # Add some black text-like content
        cv2.putText(self.test_image, 'Test', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Convert to bytes for testing
        self.image_bytes = cv2.imencode('.png', self.test_image)[1].tobytes()

    def test_preprocess_image(self):
        processed = preprocess_image(self.image_bytes)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(len(processed.shape), 2)  # Should be grayscale
        self.assertTrue(np.any(processed != self.test_image))  # Should be processed

    def test_compute_skew(self):
        # Create a rotated image
        angle = 15
        height, width = self.test_image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.test_image, rotation_matrix, (width, height))
        
        computed_angle = compute_skew(rotated)
        self.assertAlmostEqual(abs(computed_angle), angle, delta=5)

    def test_rotate_image(self):
        angle = 30
        rotated = rotate_image(self.test_image, angle)
        self.assertEqual(rotated.shape, self.test_image.shape)
        self.assertNotEqual(np.sum(rotated), np.sum(self.test_image))

if __name__ == '__main__':
    unittest.main()