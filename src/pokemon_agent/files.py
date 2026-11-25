"""Windows-safe atomic writes and file hashing."""
import hashlib
import json
import os
from pathlib import Path
import time


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


def atomic_torch_save(payload, path):
    """torch.save to a temporary file, flush it to disk, then rename it into place."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
