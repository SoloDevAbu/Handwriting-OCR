from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn
from pathlib import Path
import sys
from io import BytesIO
from PIL import Image

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent))

from preprocessing.image_processor import preprocess_image
from model.ocr_model import OCRModel
from inference.predictor import predict_text
from utils.error_handlers import (
    validation_exception_handler,
    generic_exception_handler,
    OCRException,
    ocr_exception_handler
)
from utils.logger import setup_logger, log_request
from config import settings

# Import training router
from training import router as training_router

# Initialize logger
setup_logger()

app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(OCRException, ocr_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Initialize OCR model
ocr_model = None

# Create training directory if it doesn't exist
training_dir = Path(__file__).parent.parent / "training"
training_dir.mkdir(exist_ok=True)

# Mount static files for model visualizations
app.mount("/training-files", StaticFiles(directory=str(training_dir)), name="training-files")

# Include training router
app.include_router(training_router)

@app.on_event("startup")
async def startup_event():
    global ocr_model
    try:
        ocr_model = OCRModel()
        logger.info("OCR model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OCR model: {e}")
        raise

async def validate_image(file: UploadFile) -> BytesIO:
    """Validate uploaded image file"""
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise OCRException(
            message="Invalid file type",
            details={"allowed_types": settings.ALLOWED_IMAGE_TYPES}
        )
    
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise OCRException(
            message="File too large",
            details={"max_size": settings.MAX_UPLOAD_SIZE}
        )
    
    try:
        image = Image.open(BytesIO(content))
        image.verify()  # Verify it's a valid image
        return BytesIO(content)
    except Exception as e:
        raise OCRException(message="Invalid image file")

@app.post("/api/recognize")
async def recognize_text(file: UploadFile = File(...)):
    """
    Recognize text from uploaded image
    """
    log_request("POST", "/api/recognize", {"filename": file.filename})
    
    try:
        # Validate image
        image_data = await validate_image(file)
        
        # Preprocess image
        processed_image = preprocess_image(image_data.getvalue())
        
        # Perform OCR prediction
        prediction = predict_text(ocr_model, processed_image)
        
        if not prediction:
            raise OCRException(
                message="No text detected in image",
                status_code=400
            )
        
        return {"success": True, "text": prediction}
    except OCRException as e:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Check API health status
    """
    return {
        "status": "healthy",
        "model_loaded": ocr_model is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )