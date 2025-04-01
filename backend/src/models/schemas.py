from pydantic import BaseModel, Field
from typing import Optional, List

class OCRResponse(BaseModel):
    success: bool = Field(description="Whether the OCR operation was successful")
    text: str = Field(description="Recognized text from the image")
    confidence: Optional[float] = Field(
        None,
        description="Confidence score of the recognition (0-1)",
        ge=0,
        le=1
    )
    regions: Optional[List[dict]] = Field(
        None,
        description="Detected text regions in the image"
    )

class HealthCheckResponse(BaseModel):
    status: str = Field(description="Current status of the API")
    model_loaded: bool = Field(description="Whether the OCR model is loaded and ready")
    version: str = Field(default="1.0.0", description="API version")

class ErrorResponse(BaseModel):
    message: str = Field(description="Error message")
    detail: Optional[dict] = Field(None, description="Additional error details")