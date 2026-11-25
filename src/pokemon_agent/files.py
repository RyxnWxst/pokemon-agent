"""Strict checkpoint loading and Windows-safe atomic writes."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import torch
from config import ModelConfig

# Settings that postdate older checkpoints, with the behavior those checkpoints
# were trained under. Legacy models never received move-category inputs.
LEGACY_MODEL_DEFAULTS = {"move_category_inputs": False}


def replace_file(temporary, destination):
    for attempt in range(10):
        try:
            Path(temporary).replace(destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.02 * (attempt + 1))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "readme_15token":
        raise ValueError("Expected an explicitly identified readme_15token checkpoint")
    return checkpoint


def model_config(checkpoint):
    config = checkpoint.get("model_config") or checkpoint.get("run_cfg", {}).get("model")
    if config is None:
        raise ValueError("Checkpoint is missing its model configuration")
    return ModelConfig(**{**LEGACY_MODEL_DEFAULTS, **config})


def load_weights(model, checkpoint):
    if asdict(model_config(checkpoint)) != model.reproduction_config:
        raise ValueError("Checkpoint architecture/configuration differs from the requested model")
    model.load_state_dict(checkpoint["model"], strict=True)
