"""Mode-specific FP32 AdamW; imitation freezes the critic, warmup trains it."""
import torch
from config import PPO_LR_MULTIPLIERS


def make_optimizer(model, cfg):
    zones = model.get_parameter_zones()
    groups = []
    ppo_multipliers = PPO_LR_MULTIPLIERS[cfg.ppo_lr_multipliers]
    for zone, parameters in zones.items():
        active = zone == "v" if cfg.mode == "warmup" else not (cfg.mode == "imitation" and zone == "v")
        multiplier = 1.
        if cfg.mode.startswith("ppo"):
            multiplier = ppo_multipliers[zone if zone in ("pi", "v") else "backbone"]
        decay, stable = [], []
        for name, parameter in parameters:
            parameter.requires_grad_(active)
            if active:
                excluded = "_emb" in name or "norm" in name or name.endswith(".bias") or parameter.ndim == 1
                (stable if excluded else decay).append(parameter)
        for items, weight_decay in ((decay, cfg.weight_decay), (stable, 0.)):
            if items:
                groups.append(dict(params=items, lr=cfg.lr*multiplier, weight_decay=weight_decay, name=zone))
    optimizer = torch.optim.AdamW(groups, eps=1e-5)

    def scale(step):
        if cfg.lr_schedule == "constant":
            return 1.
        if step < cfg.lr_warmup_steps:
            return (step+1)/max(1, cfg.lr_warmup_steps)
        if step < cfg.lr_warmup_steps + cfg.lr_hold_steps:
            return 1.
        progress = min(1., (step-cfg.lr_warmup_steps-cfg.lr_hold_steps)
            / max(1, cfg.lr_total_steps-cfg.lr_warmup_steps-cfg.lr_hold_steps))
        return 1./(8*progress+1)**1.5

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
