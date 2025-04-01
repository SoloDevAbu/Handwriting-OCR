import cv2
import numpy as np
from PIL import Image
import io
from loguru import logger

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess the input image for OCR.
    
    Args:
        image_bytes: Raw image bytes from the upload
    Returns:
        Preprocessed image as numpy array
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    # Deskew if needed
    angle = compute_skew(denoised)
    if abs(angle) > 0.5:
        rotated = rotate_image(denoised, angle)
    else:
        rotated = denoised
    
    # Normalize
    normalized = cv2.normalize(rotated, None, 0, 255, cv2.NORM_MINMAX)
    
    logger.info("Image preprocessing completed successfully")
    return normalized

def compute_skew(image: np.ndarray) -> float:
    """
    Compute the skew angle of the text in the image.
    
    Args:
        image: Input image
    Returns:
        Skew angle in degrees
    """
    # Find all non-zero points in the image
    coords = np.column_stack(np.where(image > 0))
    
    # Find the angle
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    
    # Handle angle range
    if angle < -45:
        angle = 90 + angle
        
    return -angle

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image by the given angle.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
    Returns:
        Rotated image
    """
    # Get image dimensions
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform rotation
    rotated = cv2.warpAffine(
        image, rotation_matrix, (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated