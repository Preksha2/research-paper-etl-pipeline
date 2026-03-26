"""
Configuration loader utility.
"""
import os
import yaml
from dotenv import load_dotenv
from loguru import logger


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {path}")
    return config


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()
    required = ["PINECONE_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}")
    logger.info("Environment variables loaded")
