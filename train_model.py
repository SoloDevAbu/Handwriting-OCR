import argparse
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'backend' / 'src'))

from model.ocr_model import OCRModel
from data.data_loader import DataLoader
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Train the OCR model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to the dataset directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to train')
    parser.add_argument('--image-size', type=int, nargs=2, default=[512, 512],
                       help='Image size for training (height width)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save the trained model')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    
    # Initialize data loader
    data_loader = DataLoader(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size)
    )
    
    # Print dataset info
    print("\nDataset Information:")
    for split, count in data_loader.data_info.items():
        print(f"{split}: {count} images")
    
    # Get datasets
    train_data = data_loader.get_training_data()
    val_data = data_loader.get_validation_data()
    test_data = data_loader.get_test_data()
    
    # Initialize and train model
    model = OCRModel()
    history = model.train(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = model.evaluate(test_data)
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    model.save_model(str(output_dir / 'final_model.h5'))
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {output_dir / 'final_model.h5'}")
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()