import numpy as np
from typing import List, Tuple, Optional
from loguru import logger
import re

from utils.error_handlers import OCRException
from utils.logger import log_model_prediction

def predict_text(model, image: np.ndarray) -> str:
    """
    Perform text prediction on an image using the OCR model.
    
    Args:
        model: The initialized OCR model
        image: Preprocessed image as numpy array
    Returns:
        Recognized text string
    Raises:
        OCRException: If there's an error during text recognition
    """
    try:
        # Log prediction attempt
        log_model_prediction(image.shape, 0)
        
        # Detect text regions in the image
        regions = model.detect_text_regions(image)
        
        if not regions:
            # If no regions detected, try processing the whole image
            text = model.recognize_text(image)
            if not text:
                raise OCRException(
                    message="No text detected in the image",
                    status_code=400,
                    details={"image_shape": image.shape}
                )
            return text
        
        # Process each detected region
        text_parts = []
        regions = merge_overlapping_regions(regions)
        
        # Log prediction with regions
        log_model_prediction(image.shape, len(regions))
        
        for region in regions:
            region_text = model.recognize_text(image, [region])
            if region_text:
                text_parts.append(region_text)
        
        if not text_parts:
            raise OCRException(
                message="Could not recognize text in detected regions",
                status_code=400,
                details={"num_regions": len(regions)}
            )
        
        # Combine and post-process the results
        combined_text = " ".join(text_parts)
        return post_process_text(combined_text)
    
    except OCRException:
        raise
    except Exception as e:
        logger.error(f"Error in text prediction: {e}")
        raise OCRException(
            message="Failed to process image",
            status_code=500,
            details={"error": str(e)}
        )

def post_process_text(text: str) -> str:
    """
    Clean and format the recognized text.
    
    Args:
        text: Raw recognized text
    Returns:
        Cleaned and formatted text
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n')
    
    # Clean up punctuation spacing
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Normalize quotes
    text = re.sub(r'[''´`]', "'", text)
    text = re.sub(r'["""]', '"', text)
    
    return text.strip()

def merge_overlapping_regions(regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Merge overlapping text regions to avoid duplicate recognition.
    
    Args:
        regions: List of text region coordinates (x1, y1, x2, y2)
    Returns:
        List of merged region coordinates
    """
    if not regions:
        return []
    
    # Sort regions by x coordinate
    sorted_regions = sorted(regions, key=lambda x: x[0])
    merged = []
    current = list(sorted_regions[0])
    
    for region in sorted_regions[1:]:
        if region[0] <= current[2]:  # Regions overlap
            current[2] = max(current[2], region[2])
            current[3] = max(current[3], region[3])
        else:
            merged.append(tuple(current))
            current = list(region)
    
    merged.append(tuple(current))
    return merged