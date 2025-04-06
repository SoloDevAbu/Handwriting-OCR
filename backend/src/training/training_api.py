from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import shutil
import zipfile
import tempfile
import json
from datetime import datetime
from loguru import logger
from typing import List, Optional
import threading
import time

from .train_model import ModelTrainer
from utils.error_handlers import OCRException

# Router for training endpoints
router = APIRouter(prefix="/api/training", tags=["training"])

# Dictionary to store active training jobs
active_trainings = {}
training_lock = threading.Lock()

class TrainingStatus:
    def __init__(self, job_id: str, data_dir: str, output_dir: str):
        self.job_id = job_id
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.status = "initializing"
        self.progress = 0
        self.start_time = datetime.now()
        self.end_time = None
        self.metrics = {}
        self.error = None

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics,
            "error": self.error
        }

def run_training(job_id: str, data_dir: str, output_dir: str, image_size: tuple, batch_size: int, epochs: int):
    """Background task to run the training process"""
    training_status = active_trainings[job_id]
    training_status.status = "processing"
    
    try:
        # Initialize model trainer
        trainer = ModelTrainer(
            data_dir=data_dir,
            output_dir=output_dir,
            image_size=image_size,
            batch_size=batch_size
        )
        
        # Prepare data
        training_status.status = "preparing_data"
        data = trainer.prepare_data()
        training_status.progress = 30
        
        # Train model
        training_status.status = "training"
        model, history = trainer.train(data, epochs=epochs)
        
        # Update training status
        training_status.status = "completed"
        training_status.progress = 100
        training_status.end_time = datetime.now()
        
        # Read metrics
        try:
            metrics_file = Path(output_dir) / "metrics" / "evaluation_results.txt"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_data = f.read()
                training_status.metrics["evaluation"] = metrics_data
            
            history_file = Path(output_dir) / "metrics" / "training_history.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    history_data = json.load(f)
                training_status.metrics["history"] = {
                    "final_accuracy": history_data["accuracy"][-1],
                    "final_val_accuracy": history_data["val_accuracy"][-1]
                }
        except Exception as e:
            logger.error(f"Error reading metrics: {e}")
            
    except Exception as e:
        logger.error(f"Training error: {e}")
        training_status.status = "failed"
        training_status.error = str(e)
        training_status.end_time = datetime.now()

@router.post("/start")
async def start_training(
    background_tasks: BackgroundTasks,
    dataset: UploadFile = File(...),
    image_size: str = Form("28,28"),
    batch_size: int = Form(32),
    epochs: int = Form(15)
):
    """
    Start a new model training job with a custom dataset.
    
    The dataset should be a ZIP file containing folders, where each folder name is the class label
    and contains images belonging to that class.
    """
    # Generate a unique job ID
    job_id = f"train_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create directories for this job
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "training" / job_id / "data"
    output_dir = base_dir / "training" / job_id / "output"
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save uploaded ZIP file
        zip_path = data_dir / "dataset.zip"
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(dataset.file, f)
        
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remove ZIP file after extraction
        os.remove(zip_path)
        
        # Parse image size
        img_width, img_height = map(int, image_size.split(","))
        image_size_tuple = (img_width, img_height)
        
        # Create training status object
        with training_lock:
            training_status = TrainingStatus(job_id, str(data_dir), str(output_dir))
            active_trainings[job_id] = training_status
        
        # Start background training task
        background_tasks.add_task(
            run_training,
            job_id=job_id,
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            image_size=image_size_tuple,
            batch_size=batch_size,
            epochs=epochs
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Training job started successfully"
        }
    
    except Exception as e:
        # Clean up directories if there was an error
        try:
            shutil.rmtree(base_dir / "training" / job_id)
        except:
            pass
        
        raise OCRException(
            message="Failed to start training job",
            details={"error": str(e)}
        )

@router.get("/status/{job_id}")
async def get_training_status(job_id: str):
    """Get the status of a training job"""
    with training_lock:
        if job_id not in active_trainings:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        status = active_trainings[job_id].to_dict()
    
    return status

@router.get("/jobs")
async def list_training_jobs():
    """List all training jobs"""
    with training_lock:
        jobs = [status.to_dict() for status in active_trainings.values()]
    
    return {"jobs": jobs}

@router.get("/models")
async def list_trained_models():
    """List all available trained models"""
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / "training"
    
    models = []
    for job_dir in models_dir.iterdir():
        if not job_dir.is_dir():
            continue
        
        output_dir = job_dir / "output"
        model_dir = output_dir / "models" / "final_model"
        
        if model_dir.exists():
            # Try to read metrics
            metrics = {}
            try:
                metrics_file = output_dir / "metrics" / "evaluation_results.txt"
                if metrics_file.exists():
                    with open(metrics_file, "r") as f:
                        metrics_text = f.read()
                        for line in metrics_text.split("\n")[:2]:  # Get first two lines (accuracy and loss)
                            if ":" in line:
                                key, value = line.split(":", 1)
                                metrics[key.strip()] = float(value.strip())
            except:
                pass
            
            # Try to read class mapping
            class_map = {}
            try:
                mapping_file = output_dir / "class_mapping.json"
                if mapping_file.exists():
                    with open(mapping_file, "r") as f:
                        class_map = json.load(f)
            except:
                pass
            
            models.append({
                "id": job_dir.name,
                "path": str(model_dir),
                "created": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
                "metrics": metrics,
                "classes": len(class_map)
            })
    
    return {"models": models}

@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: str):
    """Delete a training job and its associated files"""
    with training_lock:
        if job_id not in active_trainings:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        # Cannot delete active jobs
        if active_trainings[job_id].status in ["processing", "preparing_data", "training"]:
            raise HTTPException(status_code=400, detail="Cannot delete an active training job")
        
        # Remove job files
        base_dir = Path(__file__).parent.parent.parent
        job_dir = base_dir / "training" / job_id
        
        try:
            shutil.rmtree(job_dir)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete job files: {str(e)}")
        
        # Remove from active trainings
        del active_trainings[job_id]
    
    return {"status": "success", "message": f"Training job {job_id} deleted successfully"} 