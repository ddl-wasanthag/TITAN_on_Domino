#!/usr/bin/env python3
"""
TITAN Model Download Script for Domino Data Lab
Downloads the MahmoodLab/TITAN model to local directory
"""

import os
import torch
import logging
from pathlib import Path
from huggingface_hub import login, snapshot_download
from transformers import AutoModel
import json

# Configuration
LOCAL_MODEL_DIR = "/mnt/titan_model"
HF_MODEL_NAME = "MahmoodLab/TITAN"
HF_TOKEN_ENV = "HUGGINGFACE_TOKEN"  # Set this in Domino environment variables

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    model_dir = Path(LOCAL_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created/verified directory: {model_dir}")
    return model_dir

def authenticate_huggingface():
    """Authenticate with HuggingFace using token from environment"""
    token = os.getenv(HF_TOKEN_ENV)
    if not token:
        raise ValueError(f"HuggingFace token not found in environment variable {HF_TOKEN_ENV}")
    
    try:
        login(token=token)
        logger.info("Successfully authenticated with HuggingFace")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with HuggingFace: {e}")
        return False

def download_titan_model(local_dir):
    """Download TITAN model to local directory"""
    try:
        logger.info(f"Starting download of {HF_MODEL_NAME} to {local_dir}")
        
        # Download the model
        snapshot_download(
            repo_id=HF_MODEL_NAME,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
            resume_download=True,  # Resume if interrupted
            token=True  # Use the authenticated token
        )
        
        logger.info(f"Successfully downloaded {HF_MODEL_NAME} to {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def verify_model_download(local_dir):
    """Verify the model was downloaded correctly"""
    try:
        # Check if key files exist
        required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "modeling_titan.py"
        ]
        
        model_path = Path(local_dir)
        existing_files = list(model_path.glob("*"))
        logger.info(f"Files in model directory: {[f.name for f in existing_files]}")
        
        # Check for required files
        missing_files = []
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"Missing files: {missing_files}")
            return False
        
        # Basic file verification instead of full model loading
        # (Full loading will be done in the endpoint where paths are properly set)
        logger.info("Basic file verification successful!")
        logger.info("Note: Full model loading will be verified when starting the endpoint")
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False

def save_model_info(local_dir):
    """Save model information for endpoint usage"""
    info = {
        "model_name": HF_MODEL_NAME,
        "local_path": str(local_dir),
        "download_date": str(pd.Timestamp.now()),
        "model_type": "multimodal_pathology",
        "framework": "transformers",
        "requirements": [
            "torch==2.0.1",
            "timm==1.0.3",
            "einops==0.6.1",
            "einops-exts==0.0.4",
            "transformers==4.46.0",
            "huggingface_hub"
        ]
    }
    
    info_path = Path(local_dir) / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Model info saved to {info_path}")

def main():
    """Main download process"""
    logger.info("Starting TITAN model download process")
    
    try:
        # Setup
        model_dir = setup_directories()
        
        # Authenticate
        if not authenticate_huggingface():
            return False
        
        # Download
        if not download_titan_model(model_dir):
            return False
        
        # Verify
        if not verify_model_download(model_dir):
            logger.warning("Model verification failed, but download may still be usable")
        
        # Save info
        save_model_info(model_dir)
        
        logger.info("TITAN model download process completed successfully!")
        logger.info(f"Model available at: {model_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Download process failed: {e}")
        return False

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if not available
    success = main()
    exit(0 if success else 1)