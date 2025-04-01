from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from utils.logger import log_error
from typing import Union, Dict, Any

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors in request data"""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        })
    
    log_error(exc, f"Validation error for {request.url}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": error_details,
            "message": "Invalid request data"
        }
    )

async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any unhandled exceptions"""
    error_msg = str(exc)
    log_error(exc, f"Unhandled error for {request.url}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "An unexpected error occurred",
            "detail": error_msg if str(error_msg) else "Internal server error"
        }
    )

class OCRException(Exception):
    """Custom exception for OCR-related errors"""
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Union[Dict[str, Any], None] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)

async def ocr_exception_handler(request: Request, exc: OCRException) -> JSONResponse:
    """Handle OCR-specific exceptions"""
    log_error(exc, f"OCR error for {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.message,
            "detail": exc.details if exc.details else None
        }
    )