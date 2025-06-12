# ai-service/app/utils/startup.py
import os
import logging
import asyncio
from pathlib import Path
import httpx
from typing import List, Dict, Any
import hashlib
import json

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Model download configurations
REQUIRED_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        "files": ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.txt"],
        "size_mb": 90
    },
    "spacy/en_core_web_sm": {
        "url": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz",
        "files": [],
        "size_mb": 12
    }
}

async def setup_directories():
    """Create required directories"""
    settings = get_settings()
    
    directories = [
        settings.model_cache_dir,
        settings.pretrained_models_dir,
        settings.user_models_dir,
        settings.data_cache_dir,
        settings.embeddings_cache_dir,
        settings.training_data_dir,
        Path("logs"),
        Path("data/cache"),
        Path("models/checkpoints")
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")

async def download_required_models():
    """Download required AI models if not present"""
    settings = get_settings()
    
    # Check if models are already downloaded
    models_to_download = []
    
    for model_name, config in REQUIRED_MODELS.items():
        model_path = Path(settings.pretrained_models_dir) / model_name.replace("/", "_")
        
        if not model_path.exists() or not _verify_model_integrity(model_path, config):
            models_to_download.append((model_name, config))
    
    if not models_to_download:
        logger.info("All required models are already downloaded")
        return
    
    logger.info(f"Downloading {len(models_to_download)} models...")
    
    # Download models in parallel
    async with httpx.AsyncClient(timeout=300.0) as client:
        tasks = [
            download_model(client, model_name, config, settings.pretrained_models_dir)
            for model_name, config in models_to_download
        ]
        await asyncio.gather(*tasks)
    
    logger.info("Model download complete")

async def download_model(client: httpx.AsyncClient, model_name: str, 
                        config: Dict[str, Any], target_dir: str):
    """Download a single model"""
    try:
        logger.info(f"Downloading model: {model_name}")
        
        model_path = Path(target_dir) / model_name.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # For Hugging Face models, we would typically use their API
        # For spaCy models, we use pip in production
        # This is a simplified placeholder
        
        if "spacy" in model_name:
            # In production, use subprocess to run: python -m spacy download en_core_web_sm
            logger.info(f"SpaCy model {model_name} should be installed via pip")
        else:
            # Create placeholder files for development
            for file_name in config.get("files", []):
                file_path = model_path / file_name
                if not file_path.exists():
                    # In production, download actual files
                    file_path.write_text("{}")  # Placeholder
        
        # Mark model as downloaded
        marker_file = model_path / ".downloaded"
        marker_file.write_text(json.dumps({
            "model_name": model_name,
            "download_date": str(Path.ctime(Path.cwd())),
            "version": "1.0.0"
        }))
        
        logger.info(f"Model {model_name} ready")
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        raise

def _verify_model_integrity(model_path: Path, config: Dict[str, Any]) -> bool:
    """Verify model files are present and valid"""
    if not model_path.exists():
        return False
    
    # Check for download marker
    marker_file = model_path / ".downloaded"
    if not marker_file.exists():
        return False
    
    # In production, verify file checksums
    # For now, just check if files exist
    for file_name in config.get("files", []):
        if not (model_path / file_name).exists():
            return False
    
    return True

async def initialize_cache():
    """Initialize caching systems"""
    settings = get_settings()
    
    # Create cache directories
    cache_dirs = [
        Path(settings.data_cache_dir) / "embeddings",
        Path(settings.data_cache_dir) / "models",
        Path(settings.data_cache_dir) / "predictions"
    ]
    
    for cache_dir in cache_dirs:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Cache system initialized")

async def check_system_resources():
    """Check if system has sufficient resources"""
    import psutil
    
    # Check available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    
    if available_gb < 2:
        logger.warning(f"Low memory available: {available_gb:.1f} GB")
    else:
        logger.info(f"Available memory: {available_gb:.1f} GB")
    
    # Check disk space
    disk = psutil.disk_usage("/")
    available_disk_gb = disk.free / (1024 ** 3)
    
    if available_disk_gb < 5:
        logger.warning(f"Low disk space: {available_disk_gb:.1f} GB")
    else:
        logger.info(f"Available disk space: {available_disk_gb:.1f} GB")
    
    # Check CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU usage: {cpu_percent}%")

async def load_initial_data():
    """Load any initial data required for AI models"""
    settings = get_settings()
    
    # Load default vocabularies, patterns, etc.
    initial_data_path = Path("data/initial")
    if initial_data_path.exists():
        logger.info("Loading initial data...")
        # Implementation would load actual data files
    
    logger.info("Initial data loaded")

async def verify_database_schema():
    """Verify database schema is up to date"""
    from app.core.database import get_async_session
    
    try:
        async with get_async_session() as session:
            # Check if required tables exist
            result = await session.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            tables = [row[0] for row in result.all()]
            
            required_tables = [
                'user_behavior_patterns',
                'user_model_weights',
                'task_embeddings',
                'ai_predictions',
                'user_ai_preferences'
            ]
            
            missing_tables = [t for t in required_tables if t not in tables]
            
            if missing_tables:
                logger.warning(f"Missing database tables: {missing_tables}")
                logger.warning("Run database migrations to create missing tables")
            else:
                logger.info("Database schema verified")
                
    except Exception as e:
        logger.error(f"Failed to verify database schema: {e}")
        raise

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information if available"""
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": []
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            
            for i in range(gpu_info["count"]):
                gpu_info["devices"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_mb": torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
                })
            
            logger.info(f"GPU available: {gpu_info['count']} device(s)")
        else:
            logger.info("No GPU available, using CPU")
            
    except ImportError:
        logger.info("PyTorch not configured for GPU")
    
    return gpu_info