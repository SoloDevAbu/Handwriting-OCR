from fastapi import APIRouter

# Create a combined router for all training and inference endpoints
router = APIRouter()

try:
    from .training_api import router as training_router
    from .inference_api import router as inference_router
    
    router.include_router(training_router)
    router.include_router(inference_router)
except ImportError:
    # Module not fully set up yet
    pass

__all__ = ["router"] 