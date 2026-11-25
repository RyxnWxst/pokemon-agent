"""Action-agreement metrics over every decision, and the Wilson interval for win rates."""
import math
import torch
import torch.nn.functional as F
from pokemon_agent.rl.distributions import masked_logits


@torch.no_grad()
def action_metrics(model, dataset, batch_size=64):
    model.eval()
    observations, actions, *_ = dataset.tensorize()
    device = next(model.parameters()).device
    correct = count = 0
    loss = 0.
    groups = {name: dict(correct=0, decisions=0) for name in ("move", "switch", "tera")}
    start_mask, end_mask = model.unpacker.offsets["action_mask"]
    for start in range(0, len(actions), batch_size):
        obs = observations[start:start+batch_size].to(device)
        targets = actions[start:start+batch_size].to(device)
        logits = masked_logits(model(obs)[0], obs[:, start_mask:end_mask])
        matches = logits.argmax(-1).eq(targets)
        correct += int(matches.sum())
        count += len(targets)
        loss += float(F.cross_entropy(logits, targets, reduction="sum"))
        for name, select in (("move", targets < 4), ("switch", (targets >= 4) & (targets < 10)), ("tera", targets >= 10)):
            groups[name]["correct"] += int(matches[select].sum())
            groups[name]["decisions"] += int(select.sum())
    for group in groups.values():
        group["accuracy"] = group["correct"]/group["decisions"] if group["decisions"] else None
    return dict(correct=correct, decisions=count, accuracy=correct/count, nll=loss/count, action_groups=groups)


def combine_metrics(metrics):
    count = sum(item["decisions"] for item in metrics)
    correct = sum(item["correct"] for item in metrics)
    groups = {}
    for name in ("move", "switch", "tera"):
        n = sum(item["action_groups"][name]["decisions"] for item in metrics)
        c = sum(item["action_groups"][name]["correct"] for item in metrics)
        groups[name] = dict(correct=c, decisions=n, accuracy=c/n if n else None)
    return dict(correct=correct, decisions=count, accuracy=correct/count,
        nll=sum(item["nll"]*item["decisions"] for item in metrics)/count, action_groups=groups)


def wilson(wins, count):
    """95% Wilson score interval for a win rate of wins/count."""
    z = 1.959963984540054
    p = wins/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [center-radius, center+radius]
