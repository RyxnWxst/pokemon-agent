"""Where bundled assets live, and where runs write logs, checkpoints and data.

Assets ship inside the package. Run outputs go under the project root: the
POKEMON_AGENT_HOME environment variable if set, otherwise the working directory.
"""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
VOCAB_PATH = ASSETS_DIR / "vocab.json"


def project_root() -> Path:
    """Directory that holds logs/, checkpoints/ and data/."""
    return Path(os.environ.get("POKEMON_AGENT_HOME") or Path.cwd()).resolve()


def logs_dir() -> Path:
    return project_root() / "logs"


def checkpoints_dir() -> Path:
    return project_root() / "checkpoints"


def data_dir() -> Path:
    return project_root() / "data"


def default_seed_checkpoint() -> Path:
    """The imitation seed that training and evaluation fall back to when given no checkpoint."""
    return checkpoints_dir() / "seed" / "imitation-best.pt"
