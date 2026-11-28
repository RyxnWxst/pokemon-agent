"""Unattended experiment pipeline: baseline evaluation, then staged training from the best model.

1. Baseline: each candidate checkpoint in the plan is evaluated with ``baseline_battles``
   battles against the corrected heuristic; the better one is the model to beat.
2. Stages, in plan order. Each stage starts from the best model at the time it begins,
   runs a short critic-only warmup (the stage's reward or opponent can differ from the
   starting model's), then PPO for a fixed number of new states.
3. Every milestone is evaluated with ``battles`` battles. If it beats the best model, it is
   re-tested with ``confirm_battles`` more (a different sampling seed) and becomes the best
   only if the pooled result is higher. The best model is copied to
   checkpoints/best-experiments/best.pt and described in best.json.

The plan (logs/experiments/plan.json) is written on first start and re-read before each
stage, so later stages can be edited while the pipeline runs. Restarting the pipeline
resumes where it stopped: finished stages and evaluations are not repeated, and training
resumes from its checkpoint directory. Create logs/experiments/stop to stop cleanly.

A stage's "mode" is "ppo" (self-play), "ppo_vs_teacher" (against the heuristic), or "auto"
(the opponent setting of the stage that produced the current best model).
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from checkpoint_io import atomic_json

ROOT = Path(__file__).resolve().parent
TRAIN_STOP = ROOT / "logs/stop-training"
TRAIN_STATUS = ROOT / "logs/status.json"
LOGS = RUNS = BEST = None  # Set per workspace by Pipeline.
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)
POLL_SECONDS = 30

SHAPED_REWARD = ["--faint-reward", "fixed", "--hp-shaping", "0.5"]
DEFAULT_PLAN = {
    "server_dir": "C:/Pokemon/pokemon-showdown",
    "first_port": 8002,
    "servers": 6,
    "train_args": ["--workers", "6", "--concurrency", "192", "--matmul-precision", "high"],
    "evaluation": {"port": 8010, "battles": 1000, "confirm_battles": 1000, "baseline_battles": 2000,
                   "device": "cuda"},
    "baseline": ["checkpoints/best/ppo-readme-130m.pt",
                 "checkpoints/ppo-readme/milestones/states-000200000000.pt"],
    "milestone_every": 5_000_000,
    "stages": [
        {"name": "reward-selfplay", "mode": "ppo", "states": 40_000_000, "warmup_updates": 10,
         "reward_args": SHAPED_REWARD, "ppo_args": []},
        {"name": "vs-heuristic", "mode": "ppo_vs_teacher", "states": 30_000_000, "warmup_updates": 10,
         "reward_args": SHAPED_REWARD, "ppo_args": []},
        {"name": "lr-decay", "mode": "auto", "states": 30_000_000, "warmup_updates": 10,
         "reward_args": SHAPED_REWARD, "ppo_args": ["--lr-schedule", "linear", "--lr-warmup-steps", "100"]},
    ],
}


class StopRequested(Exception):
    pass


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {message}", flush=True)


def wilson(wins, battles, z=1.959964):
    if not battles:
        return [0., 0.]
    p = wins / battles
    denominator = 1 + z*z/battles
    centre = (p + z*z/(2*battles)) / denominator
    half = z*math.sqrt(p*(1-p)/battles + z*z/(4*battles*battles)) / denominator
    return [centre-half, centre+half]


def resolve_mode(mode, best_mode):
    if mode == "auto":
        return best_mode if best_mode in ("ppo", "ppo_vs_teacher") else "ppo"
    if mode not in ("ppo", "ppo_vs_teacher"):
        raise ValueError(f"Unknown stage mode {mode!r}")
    return mode


def warmup_args(stage, mode, start, directory):
    args = ["--mode", "warmup", "--max-updates", str(stage["warmup_updates"]), "--save-every",
            str(stage["warmup_updates"]), "--lr-schedule", "constant", "--milestone-every", "0",
            "--checkpoint-dir", str(directory), "--resume-from", str(start), *stage.get("reward_args", [])]
    return args + (["--opponent", "heuristic"] if mode == "ppo_vs_teacher" else [])


def ppo_args(stage, mode, start, directory, milestone_every):
    return ["--mode", mode, "--total-steps", str(stage["states"]), "--milestone-every", str(milestone_every),
            "--checkpoint-dir", str(directory), "--resume-from", str(start),
            *stage.get("reward_args", []), *stage.get("ppo_args", [])]


def port_in_use(port):
    with socket.socket() as check:
        return check.connect_ex(("127.0.0.1", port)) == 0


def relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


class Pipeline:
    def __init__(self, plan_path=None, workspace="experiments"):
        global LOGS, RUNS, BEST
        LOGS = ROOT / "logs" / workspace
        RUNS = ROOT / "checkpoints" / workspace
        BEST = ROOT / "checkpoints" / f"best-{workspace}"
        self.plan_path = Path(plan_path) if plan_path else LOGS / "plan.json"
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / "evals").mkdir(exist_ok=True)
        BEST.mkdir(parents=True, exist_ok=True)
        if not self.plan_path.exists():
            atomic_json(self.plan_path, DEFAULT_PLAN)
        self.plan = self.load_plan()
        self.status_path = LOGS / "status.json"
        self.status = json.loads(self.status_path.read_text()) if self.status_path.exists() else {"stages": {}}
        self.evaluations_path = LOGS / "evaluations.json"
        self.evaluations = json.loads(self.evaluations_path.read_text()) if self.evaluations_path.exists() else []
        self.best_path = BEST / "best.json"
        self.best = json.loads(self.best_path.read_text()) if self.best_path.exists() else None

    def load_plan(self):
        return json.loads(self.plan_path.read_text())

    def save_status(self, **updates):
        self.status.update(updates, updated=stamp(), best=self.best)
        if TRAIN_STATUS.exists():
            try:
                training = json.loads(TRAIN_STATUS.read_text())
                learner = training.get("learner") or {}
                self.status["training"] = dict(status=training.get("status"), mode=training.get("mode"),
                    checkpoint_dir=training.get("checkpoint_dir"), update=learner.get("update"),
                    new_states=learner.get("total_steps"), target=training.get("target_new_states"))
            except (OSError, ValueError):
                pass
        atomic_json(self.status_path, self.status)

    def check_stop(self, process=None):
        if not (LOGS / "stop").exists():
            return
        log("stop requested")
        if process is not None and process.poll() is None:
            TRAIN_STOP.parent.mkdir(exist_ok=True)
            TRAIN_STOP.touch()
            process.wait()
        (LOGS / "stop").unlink(missing_ok=True)
        raise StopRequested

    def wait_for_ports(self, ports, timeout=900):
        deadline = time.monotonic() + timeout
        while any(port_in_use(port) for port in ports):
            if time.monotonic() > deadline:
                raise RuntimeError(f"Ports still busy after {timeout}s: {[p for p in ports if port_in_use(p)]}")
            time.sleep(5)

    # ----- evaluation -------------------------------------------------------
    def evaluate(self, checkpoint, battles, seed, label):
        checkpoint = relative(checkpoint)
        for record in self.evaluations:
            if (record["checkpoint"], record["seed"], record["battles"]) == (checkpoint, seed, battles):
                return record
        evaluation = self.plan["evaluation"]
        output = LOGS / "evals" / f"{label}-seed{seed}.json"
        command = [sys.executable, "-u", str(ROOT/"run_local.py"), "--server-dir", self.plan["server_dir"],
            "--port", str(evaluation["port"]), "--skip-build", "--entry", "evaluate", "--checkpoint",
            str(ROOT/checkpoint), "--battles", str(battles), "--device", evaluation.get("device", "cuda"),
            "--seed", str(seed), "--output", str(output)]
        for attempt in (1, 2):
            self.wait_for_ports([evaluation["port"]])
            log(f"evaluating {checkpoint} ({battles} battles, seed {seed}, attempt {attempt})")
            self.save_status(phase="evaluating", evaluating=checkpoint)
            output.unlink(missing_ok=True)
            with (LOGS/"evals"/f"{label}-seed{seed}.log").open("a") as stream:
                try:
                    returncode = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        creationflags=FLAGS, timeout=4*3600).returncode
                except subprocess.TimeoutExpired:
                    returncode = "timeout"
            report = json.loads(output.read_text()) if output.exists() else {}
            if returncode == 0 and report.get("status") == "complete":
                record = dict(checkpoint=checkpoint, label=label, seed=seed, battles=report["battles"],
                    wins=report["wins"], win_rate=report["win_rate"],
                    win_rate_95pct_wilson=report["win_rate_95pct_wilson"], opponent=report.get("opponent"),
                    finished=stamp())
                self.evaluations.append(record)
                atomic_json(self.evaluations_path, self.evaluations)
                log(f"  {checkpoint}: {record['wins']}/{record['battles']} = {record['win_rate']:.3f}")
                return record
            log(f"  evaluation failed (exit {returncode}); see {output.with_suffix('.log').name}")
            time.sleep(30)
        return None

    def pooled(self, checkpoint):
        checkpoint = relative(checkpoint)
        records = [r for r in self.evaluations if r["checkpoint"] == checkpoint]
        wins, battles = sum(r["wins"] for r in records), sum(r["battles"] for r in records)
        return dict(wins=wins, battles=battles, win_rate=wins/battles if battles else 0.,
                    win_rate_95pct_wilson=wilson(wins, battles))

    def set_best(self, checkpoint, stage, mode, copy):
        score = self.pooled(checkpoint)
        history = (self.best or {}).get("history", [])
        entry = dict(checkpoint=relative(checkpoint), stage=stage, mode=mode, **score, since=stamp())
        self.best = dict(entry, copy=relative(copy) if copy else None, history=history + [entry],
            note="Win rates are pooled over every evaluation of this checkpoint against the corrected heuristic.")
        atomic_json(self.best_path, self.best)
        self.save_status()

    def baseline(self):
        if self.best:
            return
        evaluation = self.plan["evaluation"]
        scores = []
        for index, candidate in enumerate(self.plan["baseline"]):
            if self.evaluate(candidate, evaluation["baseline_battles"], 71, f"baseline-{index}") is None:
                raise RuntimeError(f"Baseline evaluation failed for {candidate}")
            scores.append((self.pooled(candidate)["win_rate"], candidate))
        rate, candidate = max(scores)
        log(f"baseline to beat: {candidate} at {rate:.3f}")
        self.set_best(candidate, "baseline", "ppo", copy=None)

    def consider(self, milestone, stage, mode):
        evaluation = self.plan["evaluation"]
        label = f"{stage}-{milestone.stem}"
        first = self.evaluate(milestone, evaluation["battles"], 71, label)
        if first is None or first["win_rate"] <= self.best["win_rate"]:
            return
        log(f"  {milestone.name} beat the best on first test ({first['win_rate']:.3f} > "
            f"{self.best['win_rate']:.3f}); confirming")
        if self.evaluate(milestone, evaluation["confirm_battles"], 72, label) is None:
            return
        score = self.pooled(milestone)
        if score["win_rate"] <= self.best["win_rate"]:
            log(f"  not confirmed: pooled {score['win_rate']:.3f}")
            return
        temporary = BEST / "best.pt.tmp"
        shutil.copyfile(milestone, temporary)
        os.replace(temporary, BEST / "best.pt")
        self.set_best(milestone, stage, mode, copy=BEST/"best.pt")
        log(f"  NEW BEST {relative(milestone)}: pooled {score['wins']}/{score['battles']} = {score['win_rate']:.3f}")

    # ----- training ---------------------------------------------------------
    def train(self, args, log_name, on_poll=None):
        ports = [self.plan["first_port"] + i for i in range(self.plan["servers"])]
        command = [sys.executable, "-u", str(ROOT/"run_local.py"), "--server-dir", self.plan["server_dir"],
            "--port", str(self.plan["first_port"]), "--servers", str(self.plan["servers"]), "--skip-build",
            *self.plan["train_args"], *args]
        for attempt in (1, 2, 3):
            self.wait_for_ports(ports)
            log(f"training ({log_name}, attempt {attempt}): {' '.join(args)}")
            with (LOGS/f"{log_name}.log").open("a") as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=FLAGS)
                while process.poll() is None:
                    self.check_stop(process)
                    if on_poll:
                        on_poll()
                    self.save_status()
                    time.sleep(POLL_SECONDS)
            self.check_stop()
            if process.returncode == 0:
                return
            log(f"  training exited with {process.returncode}; retrying from its checkpoint directory")
            time.sleep(60)
        raise RuntimeError(f"{log_name} failed three times; see {LOGS/(log_name+'.log')}")

    def run_stage(self, stage):
        name = stage["name"]
        state = self.status["stages"].setdefault(name, {})
        if state.get("done"):
            return
        if "start" not in state:
            state.update(start=self.best["checkpoint"], started=stamp(),
                mode=resolve_mode(stage["mode"], self.best.get("mode")), evaluated=[])
        mode, start = state["mode"], ROOT/state["start"]
        self.save_status(stage=name)
        warmup_dir, ppo_dir = RUNS/name/"warmup", RUNS/name/"ppo"
        if stage.get("warmup_updates") and not state.get("warmup_done"):
            self.save_status(phase="warmup")
            self.train(warmup_args(stage, mode, start, warmup_dir), f"{name}-warmup")
            state["warmup_done"] = True
            self.save_status()
        ppo_start = sorted(warmup_dir.glob("learner_update_*.pt"))[-1] if stage.get("warmup_updates") else start

        def evaluate_new_milestones():
            for milestone in sorted((ppo_dir/"milestones").glob("states-*.pt")):
                if milestone.name not in state["evaluated"]:
                    self.consider(milestone, name, mode)
                    state["evaluated"].append(milestone.name)
                    self.save_status(phase="ppo")

        self.save_status(phase="ppo")
        self.train(ppo_args(stage, mode, ppo_start, ppo_dir, self.plan["milestone_every"]), f"{name}-ppo",
            on_poll=evaluate_new_milestones)
        evaluate_new_milestones()
        state.update(done=True, finished=stamp())
        self.save_status()
        log(f"stage {name} finished; best is {self.best['checkpoint']} at {self.best['win_rate']:.3f}")

    def run(self):
        try:
            self.save_status(status="running", phase="baseline", pid=os.getpid())
            self.baseline()
            index = 0
            while True:
                stages = self.load_plan()["stages"]
                if index >= len(stages):
                    break
                self.plan = self.load_plan()
                self.run_stage(stages[index])
                index += 1
            self.save_status(status="complete", phase="done")
            log("pipeline complete")
        except StopRequested:
            self.save_status(status="stopped")
        except BaseException as exc:
            self.save_status(status="failed", error=repr(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", help="Plan file; defaults to logs/<workspace>/plan.json")
    parser.add_argument("--workspace", default="experiments",
        help="Names logs/<workspace>, checkpoints/<workspace> and checkpoints/best-<workspace>")
    args = parser.parse_args()
    Pipeline(args.plan, args.workspace).run()


if __name__ == "__main__":
    main()
