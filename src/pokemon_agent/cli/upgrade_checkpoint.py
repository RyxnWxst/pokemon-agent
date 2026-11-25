"""Convert a pre-category checkpoint into one that receives README-era move categories.

The upgraded weights reproduce the source policy and value outputs; the restored
Physical/Special/Status inputs start with zero weight and are learned by further
imitation (imitation.py --init-checkpoint). The source file is never modified.
"""
import argparse
import hashlib
from pathlib import Path
import torch
from checkpoint_io import atomic_json, load_weights, model_config, read_checkpoint, replace_file
from config import ROOT
from model import make_model, upgrade_move_category_inputs


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_observations(path, limit):
    """Accept the test fixture format or an imitation episode list."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        observations = data["observations"]
    else:
        observations, count = [], 0
        for episode in data:
            observations.append(episode[0])
            count += len(episode[0])
            if count >= limit:
                break
        observations = torch.cat(observations)
    return observations[:limit].float()


@torch.no_grad()
def outputs(model, observations, batch_size=256):
    parts = [model(observations[start:start+batch_size])[:3] for start in range(0, len(observations), batch_size)]
    return [torch.cat(items) for items in zip(*parts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--observations", default=str(ROOT/"tests/fixtures/battle-observations.pt"),
        help="Real observations for the equality check: fixture file or imitation episode list")
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.exists() or output.with_suffix(".json").exists():
        parser.error("Output already exists; choose a new path")
    torch.set_num_threads(2)
    checkpoint = read_checkpoint(source)
    legacy = make_model(model_config(checkpoint)).eval()
    load_weights(legacy, checkpoint)
    if legacy.move_category_inputs:
        parser.error("Source checkpoint already uses move-category inputs")
    upgraded = upgrade_move_category_inputs(legacy).eval()

    observations = load_observations(args.observations, args.max_decisions)
    adapted = upgraded.adapt_observations(observations)
    category_inputs = int((adapted[:, upgraded.category_destination] > 0).sum())
    if category_inputs == 0:
        parser.error("Verification observations contain no revealed move categories")
    before, after = outputs(legacy, observations), outputs(upgraded, observations)
    differences = {name: float((old - new).abs().max())
        for name, old, new in zip(("logits", "value_logits", "values"), before, after)}
    if max(differences.values()) > args.tolerance:
        raise RuntimeError(f"Upgraded outputs differ from the source: {differences}")
    start, end = upgraded.unpacker.offsets["action_mask"]
    mask = observations[:, start:end] > .5
    agreement = float((before[0].masked_fill(~mask, -1e9).argmax(1)
        == after[0].masked_fill(~mask, -1e9).argmax(1)).float().mean())

    report = dict(status="complete", kind="move_category_inputs", source=str(source),
        source_sha256=sha256(source), output=str(output),
        zeroed_move_net_input_columns=upgraded.category_weight_columns(),
        verification=dict(observations=str(Path(args.observations).resolve()), decisions=len(observations),
            category_inputs_present=category_inputs, max_abs_difference=differences,
            argmax_agreement=agreement, tolerance=args.tolerance))
    payload = dict(project="ps-ppo-15token-upgrade", architecture="readme_15token", format_version=1,
        phase=checkpoint.get("phase", "imitation"), model=upgraded.state_dict(),
        model_config=upgraded.reproduction_config, upgrade=report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    replace_file(temporary, output)
    report["output_sha256"] = sha256(output)
    atomic_json(output.with_suffix(".json"), report)
    print(report)


if __name__ == "__main__":
    main()
