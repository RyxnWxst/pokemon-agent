"""Slice a flat observation tensor back into named, reshaped blocks."""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn


class ObservationUnpacker(nn.Module):
    """
    Slices and reshapes flat observation tensors into structured components.
    
    Expects metadata containing 'offsets' and dynamic dimensions from the 
    ObservationAssembler.
    """
    def __init__(self, meta: dict):
        super().__init__()
        self.meta = meta
        self.offsets = meta["offsets"]
        self.dim_body = meta["dim_pokemon_body"]
        self.dim_move_sc = meta["dim_move_scalars"] // 4
        self.dim_transition_sc = meta["dim_transition_scalars"]
        self.transition_map = meta["feature_map"]["transition"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {name: x[:, start:end] for name, (start, end) in self.offsets.items()}
        
        # Reshape structured fields
        out["pokemon_body"] = out["pokemon_body"].reshape(-1, 12, self.dim_body)
        out["pokemon_ids"]  = out["pokemon_ids"].reshape(-1, 12, 2).to(torch.long)
        out["ability_ids"]  = out["ability_ids"].reshape(-1, 12, 4).to(torch.long)
        out["move_ids"]     = out["move_ids"].reshape(-1, 12, 4).to(torch.long)
        out["move_scalars"] = out["move_scalars"].reshape(-1, 12, 4, self.dim_move_sc)
        out["transition_ids"] = out["transition_ids"].reshape(-1, self.meta["dim_transition_ids"]).to(torch.long)
        out["transition_scalars"] = out["transition_scalars"].reshape(-1, self.dim_transition_sc)
        
        tmap = self.transition_map
        out["transition_move_ids"] = out["transition_ids"][:, tmap["move_ids"][0]:tmap["move_ids"][1]]
        out["transition_pokemon_ids"] = out["transition_ids"][:, tmap["pokemon_ids"][0]:tmap["pokemon_ids"][1]]
        out["transition_ability_ids"] = out["transition_ids"][:, tmap["ability_ids"][0]:tmap["ability_ids"][1]]
        out["transition_item_ids"] = out["transition_ids"][:, tmap["item_ids"][0]:tmap["item_ids"][1]]
        
        return out
