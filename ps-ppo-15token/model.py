"""README-era architecture with explicit adapters for the corrected dataset.

The historical model/config copies are byte-for-byte from de8cc69. This wrapper
adapts input layout and the current runner interface only; it preserves the
15-token architecture, move aggregation, initialization, masks, and readout.
"""
from dataclasses import asdict
import torch
from obs_assembler import ObservationAssembler
from ppo_core import ObservationUnpacker
from vendor.historical_core import PokeTransformer
from config import ModelConfig

# README-era move scalars stored the Physical/Special/Status one-hot at indices
# 3-5, ahead of the priority one-hot at 6-18. The current schema leaves 3-5 empty
# and stores the category after the type one-hot, outside the slice read by the
# historical move encoder.
HISTORICAL_CATEGORY_START = 3
CATEGORY_COUNT = 3


def category_columns(metadata):
    """Flat observation columns for the historical category slots and their current source."""
    move_map = metadata["feature_map"]["move"]
    first, last = move_map["category_raw"]
    if tuple(move_map["onehots_raw"]) != (3, 19) or last - first != CATEGORY_COUNT + 1:
        raise ValueError("Unexpected move schema; cannot restore historical category inputs")
    start, end = metadata["offsets"]["move_scalars"]
    width = metadata["dim_move_scalars"] // 4
    slots = range(start, end, width)
    destination = [slot + HISTORICAL_CATEGORY_START + i for slot in slots for i in range(CATEGORY_COUNT)]
    # category_raw index 0 is unknown; physical, special and status follow in vocabulary order.
    source = [slot + first + 1 + i for slot in slots for i in range(CATEGORY_COUNT)]
    return torch.tensor(destination), torch.tensor(source)


class ReadmePokeTransformer(PokeTransformer):
    def __init__(self, metadata=None, model_config=None):
        metadata = metadata or ObservationAssembler().meta
        model_config = model_config or ModelConfig()
        self.reproduction_config = asdict(model_config)
        super().__init__(act_dim=14, meta=metadata, emb_dims=model_config.emb_dims,
            out_dims=model_config.out_dims, bank_dims=model_config.bank_dims,
            bank_ranges=model_config.bank_ranges, n_heads=model_config.n_heads,
            n_layers=model_config.n_layers, ff_expansion=model_config.ff_expansion,
            v_bins=51, dropout=model_config.dropout)
        # The current observations combine transition IDs and include more than
        # ten transition scalars. This unpacker uses their actual metadata.
        self.unpacker = ObservationUnpacker(metadata)
        self.move_category_inputs = bool(getattr(model_config, "move_category_inputs", False))
        destination, source = category_columns(metadata)
        # Index constants, not learned state: excluded from checkpoints.
        self.register_buffer("category_destination", destination, persistent=False)
        self.register_buffer("category_source", source, persistent=False)

    def _calc_pok_in(self, emb, bank, out):
        body = self.f_map["body"]
        stat_count = body["stats_int"][1] - body["stats_int"][0]
        raw_count = body["boosts_raw"][1] - body["boosts_raw"][0]
        raw_count += self.meta["dim_pokemon_body"] - body["flags_raw"][0]
        # Current input stores 15 stat-bound values versus six historical stats.
        return (emb["pokemon"] + emb["item"] + bank["val_100"]*2
            + bank["stat"]*(stat_count+2) + out["ability_vec"]
            + self.meta["n_move_slots"]*out["move_vec"] + raw_count)

    def category_weight_columns(self):
        """move_net input columns fed by the historical category one-hot."""
        offset = (self.move_emb.embedding_dim + 2*self.val_100_emb.embedding_dim
            + self.power_emb.embedding_dim
            + HISTORICAL_CATEGORY_START - self.f_map["move"]["onehots_raw"][0])
        return list(range(offset, offset + CATEGORY_COUNT))

    def adapt_observations(self, obs_flat):
        """Map current observations onto the historical encoder's input layout."""
        adapted = obs_flat.float().clone()
        # The historical encoder expects turn/100; current data stores raw turns.
        turn = self.meta["offsets"]["global_scalars"][0]
        adapted[:, turn] /= 100.
        if self.move_category_inputs:
            adapted[:, self.category_destination] = adapted[:, self.category_source]
        return adapted

    def forward(self, obs_flat, episode_ids=None, batch_seq_len=0, **kwargs):
        logits, values, expected = super().forward(self.adapt_observations(obs_flat))
        return logits, values, expected, None, None

    def get_parameter_zones(self):
        """Assign every parameter exactly once; warmup trains the critic only."""
        zones = {name: [] for name in ("embeddings", "subnets", "transformer", "readout", "pi", "v")}
        for name, parameter in self.named_parameters():
            if name == "actor_tok" or name.startswith("pi_head."):
                zone = "pi"
            elif name == "critic_tok" or name.startswith("v_head."):
                zone = "v"
            elif name.startswith("transformer."):
                zone = "transformer"
            elif name.startswith("readout_"):
                zone = "readout"
            elif "_emb." in name:
                zone = "embeddings"
            else:
                zone = "subnets"
            zones[zone].append((name, parameter))
        return zones


def make_model(model_config=None):
    return ReadmePokeTransformer(model_config=model_config)


def upgrade_move_category_inputs(legacy):
    """Return a category-enabled copy of a legacy model with identical outputs.

    A legacy move encoder only ever received zeros in the historical category
    columns, so those weights never affected its outputs. Zeroing them keeps the
    upgraded policy and value unchanged while later training learns to use the
    restored category inputs.
    """
    if legacy.move_category_inputs:
        raise ValueError("Model already receives move-category inputs")
    config = ModelConfig(**{**legacy.reproduction_config, "move_category_inputs": True})
    upgraded = ReadmePokeTransformer(metadata=legacy.meta, model_config=config)
    upgraded.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        upgraded.move_net[0].weight[:, upgraded.category_weight_columns()] = 0.
    return upgraded.to(next(legacy.parameters()).device)
