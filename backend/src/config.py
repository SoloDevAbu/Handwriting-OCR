from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Handwriting OCR API"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]
    
    # Model Settings
    MODEL_PATH: Optional[str] = None
    CONFIDENCE_THRESHOLD: float = 0.5
    MAX_IMAGE_SIZE: int = 2048  # Maximum image dimension
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/tiff"]
    
    # Performance
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Create global settings instance
settings = Settings()

# Project Paths
ROOT_DIR = Path(__file__).parent.parent
MODEL_DIR = ROOT_DIR / "models"
UPLOAD_DIR = ROOT_DIR / "uploads"
LOG_DIR = ROOT_DIR / "logs"

# Create necessary directories
for dir_path in [MODEL_DIR, UPLOAD_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True)