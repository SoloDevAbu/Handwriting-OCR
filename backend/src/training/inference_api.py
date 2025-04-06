from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from loguru import logger
from pathlib import Path
import os

from .model_inference import CustomModelInference, get_available_models, load_model
from utils.error_handlers import OCRException

# Router for inference endpoints
router = APIRouter(prefix="/api/custom-models", tags=["custom-models"])

# Cache for loaded models
loaded_models = {}

@router.get("/")
async def list_models():
    """List all available custom trained models"""
    try:
        models = get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise OCRException(
            message="Failed to list models",
            details={"error": str(e)}
        )

@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_id: str = Form(...),
):
    """
    Run inference on an image using a custom trained model.
    
    Args:
        file: Image file to predict
        model_id: ID of the model to use
    """
    try:
        # Load model if not already in cache
        model = loaded_models.get(model_id)
        if not model:
            model = load_model(model_id)
            loaded_models[model_id] = model
        
        # Read image data
        image_data = await file.read()
        
        # Run prediction
        prediction = model.predict(image_data)
        
        return {
            "success": True,
            "prediction": prediction,
            "model_id": model_id
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise OCRException(
            message="Prediction failed",
            details={"error": str(e)}
        )

@router.post("/batch-predict")
async def batch_predict(
    files: list[UploadFile] = File(...),
    model_id: str = Form(...),
):
    """
    Run inference on multiple images using a custom trained model.
    
    Args:
        files: List of image files to predict
        model_id: ID of the model to use
    """
    try:
        # Load model
        model = loaded_models.get(model_id)
        if not model:
            model = load_model(model_id)
            loaded_models[model_id] = model
        
        # Process each image
        results = []
        for file in files:
            try:
                # Read image data
                image_data = await file.read()
                
                # Run prediction
                prediction = model.predict(image_data)
                
                results.append({
                    "filename": file.filename,
                    "prediction": prediction
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "results": results,
            "model_id": model_id
        }
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise OCRException(
            message="Batch prediction failed",
            details={"error": str(e)}
        )

@router.get("/model-info/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model"""
    try:
        # Get base directory
        base_dir = Path(__file__).parent.parent.parent
        model_dir = base_dir / "training" / model_id
        
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model not found with ID: {model_id}")
        
        # Check for model files
        model_path = model_dir / "output" / "models" / "final_model"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model files not found for ID: {model_id}")
        
        # Get class mapping
        class_mapping = {}
        mapping_file = model_dir / "output" / "class_mapping.json"
        if mapping_file.exists():
            import json
            with open(mapping_file, "r") as f:
                class_mapping = json.load(f)
        
        # Get model metrics
        metrics = {}
        metrics_file = model_dir / "output" / "metrics" / "evaluation_results.txt"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics_text = f.read()
                metrics["evaluation"] = metrics_text
        
        # Get model history
        history = {}
        history_file = model_dir / "output" / "metrics" / "training_history.json"
        if history_file.exists():
            import json
            with open(history_file, "r") as f:
                history = json.load(f)
        
        # Get additional info
        created_date = model_path.stat().st_mtime
        from datetime import datetime
        created = datetime.fromtimestamp(created_date).isoformat()
        
        # Check for plot images
        plots = []
        plots_dir = model_dir / "output" / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))
            plots = [str(p.relative_to(base_dir)) for p in plot_files]
        
        return {
            "id": model_id,
            "path": str(model_path),
            "created": created,
            "class_mapping": class_mapping,
            "num_classes": len(class_mapping),
            "metrics": metrics,
            "history": history,
            "plots": plots
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise OCRException(
            message="Failed to get model information",
            details={"error": str(e)}
        ) 