# Handwriting OCR System

A full-stack application that performs Optical Character Recognition (OCR) on handwritten text using machine learning.

## Features

- Upload images through drag-and-drop or file selection
- Real-time text recognition from handwritten content
- Modern, responsive web interface
- Advanced image preprocessing pipeline
- Hybrid OCR approach using TensorFlow and Tesseract

## System Architecture

### Frontend
- Next.js 14 with TypeScript
- React components for image upload and display
- Tailwind CSS for styling
- Axios for API communication

### Backend
- FastAPI server with async support
- TensorFlow & Tesseract for OCR
- OpenCV for image preprocessing
- Comprehensive error handling and logging

## Prerequisites

- Python 3.8+
- Node.js 18+
- Tesseract OCR installed on the system
- CUDA-compatible GPU (optional, for better performance)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd handwriting-ocr-project
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix:
source venv/bin/activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

## Running the Application

1. Start the backend server:
```bash
cd backend
# Activate virtual environment if not already activated
python src/main.py
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Access the application at `http://localhost:3000`

## Project Structure

```
handwriting-ocr-project/
├── frontend/               # Next.js frontend application
├── backend/               # FastAPI backend server
│   ├── src/
│   │   ├── preprocessing/ # Image preprocessing modules
│   │   ├── model/        # ML model definition
│   │   ├── inference/    # Text recognition inference
│   │   └── utils/        # Helper functions
│   └── requirements.txt
├── tests/                # Test suites
└── notebooks/           # Jupyter notebooks for exploration
```

## API Endpoints

- `POST /api/recognize`: Upload an image for OCR recognition
  - Accepts: multipart/form-data with 'file' field
  - Returns: JSON with recognized text

## Future Improvements

1. Model improvements:
   - Fine-tune on custom datasets
   - Add support for different languages
   - Implement more advanced preprocessing techniques

2. Features to add:
   - Batch processing support
   - Real-time recognition from camera
   - Export results in different formats
   - User authentication system

3. Performance optimizations:
   - Model quantization
   - Caching system
   - Load balancing for scale

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.