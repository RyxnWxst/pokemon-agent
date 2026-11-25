"""Strict checkpoint reading: architecture checks, legacy defaults, and loading a model from a file."""
from dataclasses import asdict
import torch
from pokemon_agent.config import ModelConfig

# Settings that postdate older checkpoints, with the behavior those checkpoints
# were trained under. Legacy models never received move-category inputs.
LEGACY_MODEL_DEFAULTS = {"move_category_inputs": False}


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


def load_model(path_or_checkpoint, device="cpu"):
    """Build the checkpoint's architecture, load its weights, and return it in eval mode."""
    from pokemon_agent.model.transformer import make_model

    checkpoint = (read_checkpoint(path_or_checkpoint) if not isinstance(path_or_checkpoint, dict)
        else path_or_checkpoint)
    model = make_model(model_config(checkpoint))
    load_weights(model, checkpoint)
    return model.to(device).eval()
