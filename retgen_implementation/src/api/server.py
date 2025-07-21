#!/usr/bin/env python3
"""FastAPI REST API server for RETGEN model."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.core.config import RETGENConfig
from run_retgen import RETGENSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)

# Global model instance
model_instance: Optional[RETGENSystem] = None
model_lock = asyncio.Lock()


class GenerationRequest(BaseModel):
    """Text generation request."""
    prompt: str = Field(..., description="Input prompt for text generation")
    max_length: int = Field(50, ge=1, le=500, description="Maximum generation length")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    num_generations: int = Field(1, ge=1, le=5, description="Number of generations")


class GenerationResponse(BaseModel):
    """Text generation response."""
    prompt: str
    generations: List[str]
    metadata: Dict[str, Any]


class TrainingRequest(BaseModel):
    """Model training request."""
    documents: List[str] = Field(..., description="Training documents")
    validation_split: float = Field(0.1, ge=0.0, le=0.3, description="Validation split ratio")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Config overrides")


class TrainingResponse(BaseModel):
    """Training response."""
    success: bool
    metrics: Dict[str, Any]
    message: str


class ModelInfo(BaseModel):
    """Model information."""
    is_loaded: bool
    is_trained: bool
    config: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    version: str = "1.0.0"


# Create FastAPI app
app = FastAPI(
    title="RETGEN API",
    description="Retrieval-Enhanced Text Generation REST API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup if MODEL_PATH is set."""
    model_path = os.environ.get("RETGEN_MODEL_PATH")
    if model_path:
        try:
            await load_model(Path(model_path))
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model_instance is not None
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if model_instance is None:
        return ModelInfo(
            is_loaded=False,
            is_trained=False,
            config=None,
            metrics=None
        )
    
    return ModelInfo(
        is_loaded=True,
        is_trained=model_instance.is_trained,
        config=model_instance.config.__dict__ if model_instance.config else None,
        metrics=model_instance.training_metrics if model_instance.is_trained else None
    )


@app.post("/model/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a new RETGEN model."""
    global model_instance
    
    async with model_lock:
        try:
            # Create config with overrides
            config = RETGENConfig(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dim=384,
                device="cpu",
                use_gpu=False
            )
            
            if request.config_overrides:
                for key, value in request.config_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Create new model instance
            model_instance = RETGENSystem(config)
            
            # Split documents
            num_val = int(len(request.documents) * request.validation_split)
            val_docs = request.documents[:num_val] if num_val > 0 else None
            train_docs = request.documents[num_val:] if num_val > 0 else request.documents
            
            # Train in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(
                executor,
                model_instance.train,
                train_docs,
                val_docs
            )
            
            return TrainingResponse(
                success=True,
                metrics=metrics,
                message=f"Model trained successfully on {len(train_docs)} documents"
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/load")
async def load_model(model_path: Path):
    """Load a pre-trained model."""
    global model_instance
    
    async with model_lock:
        try:
            model_instance = RETGENSystem.load(model_path)
            return {"success": True, "message": f"Model loaded from {model_path}"}
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/save")
async def save_model(model_path: Path):
    """Save the current model."""
    if model_instance is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    if not model_instance.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")
    
    async with model_lock:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, model_instance.save, model_path)
            return {"success": True, "message": f"Model saved to {model_path}"}
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from prompt."""
    if model_instance is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    if not model_instance.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")
    
    try:
        # Generate in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        generations = []
        
        for _ in range(request.num_generations):
            text = await loop.run_in_executor(
                executor,
                model_instance.generate,
                request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k
            )
            generations.append(text)
        
        return GenerationResponse(
            prompt=request.prompt,
            generations=generations,
            metadata={
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "model_config": {
                    "embedding_model": model_instance.config.embedding_model,
                    "retrieval_k": model_instance.config.retrieval_k
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/batch")
async def generate_batch(prompts: List[str], **kwargs):
    """Generate text for multiple prompts."""
    if model_instance is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    if not model_instance.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")
    
    try:
        loop = asyncio.get_event_loop()
        tasks = []
        
        for prompt in prompts:
            task = loop.run_in_executor(
                executor,
                model_instance.generate,
                prompt,
                **kwargs
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        return {
            "prompts": prompts,
            "generations": results,
            "metadata": kwargs
        }
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()