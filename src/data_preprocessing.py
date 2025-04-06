import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def preprocess_image(image_path, target_size=(28, 28)):
    """
    Preprocess a single image:
    1. Read the image
    2. Convert to grayscale
    3. Resize to target size
    4. Normalize pixel values
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize image
    resized = cv2.resize(gray, target_size)
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    return normalized

def process_dataset(input_dir, output_dir):
    """
    Process all images in the input directory and save processed images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions)]
    
    processed_images = []
    labels = []
    
    for img_file in image_files:
        try:
            # Process image
            img_path = os.path.join(input_dir, img_file)
            processed_img = preprocess_image(img_path)
            
            # Save processed image
            output_path = os.path.join(output_dir, f"processed_{img_file}")
            cv2.imwrite(output_path, processed_img * 255)
            
            processed_images.append(processed_img)
            labels.append(img_file.split('_')[0])  # Assuming filename format: label_image.jpg
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(processed_images)
    y = np.array(labels)
    
    # Save processed data
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    
    return X, y

if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    # Process handwritten text dataset
    print("Processing handwritten text dataset...")
    handwritten_raw = raw_dir / 'handwritten'
    handwritten_processed = processed_dir / 'handwritten'
    X_handwritten, y_handwritten = process_dataset(handwritten_raw, handwritten_processed)
    
    # Process key dataset
    print("Processing key dataset...")
    key_raw = raw_dir / 'keys'
    key_processed = processed_dir / 'keys'
    X_keys, y_keys = process_dataset(key_raw, key_processed)
    
    print("Data preprocessing completed!") 