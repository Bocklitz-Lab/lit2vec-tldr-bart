# src/utils.py

import os
import random
import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# -----------------
# Reproducibility
# -----------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across:
      - Python's random
      - NumPy
      - PyTorch (CPU & GPU)

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Reproducibility seed set to {seed}")


# -----------------
# Logging
# -----------------

def setup_logger(name: str = "lit2vec", level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger with a clean format.

    Args:
        name: logger name
        level: logging level (e.g. logging.DEBUG, logging.INFO)

    Returns:
        logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers in Jupyter/colab
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# -----------------
# Filesystem / IO
# -----------------

def save_json(obj: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """
    Save a dictionary to JSON.

    Args:
        obj: dict to save
        path: output path
        indent: indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        path: path to JSON file

    Returns:
        dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists.

    Args:
        path: directory path

    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------
# Device Helpers
# -----------------

def get_device(force_cpu: bool = False) -> torch.device:
    """
    Return best available torch device.

    Args:
        force_cpu: force returning CPU

    Returns:
        torch.device ("cuda", "mps" or "cpu")
    """
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name(device: Optional[torch.device] = None) -> str:
    """
    Human-readable device name.
    """
    if device is None:
        device = get_device()
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type
