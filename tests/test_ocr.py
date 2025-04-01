import unittest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'backend' / 'src'))

from model.ocr_model import OCRModel
from inference.predictor import predict_text, post_process_text

class TestOCR(unittest.TestCase):
    def setUp(self):
        # Create a mock model for testing
        self.mock_model = Mock(spec=OCRModel)
        self.test_image = np.ones((100, 100), dtype=np.uint8) * 255

    @patch('model.ocr_model.hub')
    def test_ocr_model_initialization(self, mock_hub):
        # Test model initialization
        model = OCRModel()
        self.assertIsNotNone(model)
        mock_hub.load.assert_called_once()

    def test_post_process_text(self):
        # Test text post-processing
        test_cases = [
            ("Hello   World  ", "Hello World"),
            ("Test123! @#$", "Test123!"),
            ("Line1\n  Line2  \nLine3", "Line1 Line2 Line3"),
            ("Mixed Case   text", "Mixed Case text"),
        ]
        
        for input_text, expected in test_cases:
            result = post_process_text(input_text)
            self.assertEqual(result, expected)

    def test_predict_text_with_regions(self):
        # Mock the model's text recognition
        self.mock_model.detect_text_regions.return_value = [(0, 0, 50, 50)]
        self.mock_model.recognize_text.return_value = "Test Text"
        
        result = predict_text(self.mock_model, self.test_image)
        self.assertEqual(result, "Test Text")
        self.mock_model.detect_text_regions.assert_called_once()
        self.mock_model.recognize_text.assert_called_once()

    def test_predict_text_error_handling(self):
        # Test error handling during prediction
        self.mock_model.detect_text_regions.side_effect = Exception("Test error")
        
        with self.assertRaises(Exception):
            predict_text(self.mock_model, self.test_image)

if __name__ == '__main__':
    unittest.main()