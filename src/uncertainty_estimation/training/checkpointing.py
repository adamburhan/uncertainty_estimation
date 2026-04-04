"""Utilities for discovering, parsing, and loading checkpoints.

Understands the two filename patterns produced by trainer.py:
  - {exp_name}_best_epoch={E}_loss={L:.4f}.pth
  - {exp_name}_epoch_{EEEE}.pth
  - {exp_name}_metrics.pth
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Union

import torch

# Regex patterns for the two checkpoint types.
# Groups: (epoch, loss)
BEST_REGEX = r"_best_epoch=(\d+)_loss=([\d.eE+-]+)\.pth$"
# Groups: (epoch,)
PERIODIC_REGEX = r"_epoch_(\d+)\.pth$"


def _sorted_nicely(l: List[str]) -> List[str]:
    """Sort strings the way humans expect (numeric substrings compared as ints)."""
    def _key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]
    return sorted(l, key=_key)


def extract_metrics(file_name: str, exp_name: Optional[str] = None) -> Dict:
    """Parse checkpoint metadata from its filename.

    Returns a dict with keys:
        type:  "best" | "periodic"
        epoch: int
        loss:  float (only for "best" checkpoints, else None)

    Returns None if the filename doesn't match any known pattern.
    """
    prefix = re.escape(exp_name) if exp_name is not None else ".*"

    # Try best checkpoint pattern first
    m = re.match(f"^{prefix}{BEST_REGEX}", file_name)
    if m:
        i = 1 if exp_name is not None else 2  # group offset when exp_name is ".*"
        # With a concrete prefix the groups start at 1; with .* they shift by 1
        groups = m.groups()
        if exp_name is None:
            epoch, loss = int(groups[1]), float(groups[2])
        else:
            epoch, loss = int(groups[0]), float(groups[1])
        return {"type": "best", "epoch": epoch, "loss": loss}

    # Try periodic checkpoint pattern
    m = re.match(f"^{prefix}{PERIODIC_REGEX}", file_name)
    if m:
        groups = m.groups()
        epoch = int(groups[-1])
        return {"type": "periodic", "epoch": epoch, "loss": None}

    return None


def get_model_files(
    checkpoint_dir: str,
    exp_name: Optional[str] = None,
    kind: Optional[str] = None,
) -> List[str]:
    """List checkpoint files in *checkpoint_dir*, optionally filtered.

    Args:
        checkpoint_dir: Directory containing .pth files.
        exp_name: Filter to this experiment name. If None, match all.
        kind: "best", "periodic", or None (both).

    Returns:
        Naturally-sorted list of matching filenames (basenames, not full paths).
    """
    prefix = re.escape(exp_name) if exp_name is not None else ".*"

    patterns = []
    if kind in (None, "best"):
        patterns.append(re.compile(f"^{prefix}{BEST_REGEX}"))
    if kind in (None, "periodic"):
        patterns.append(re.compile(f"^{prefix}{PERIODIC_REGEX}"))

    matches = []
    for f in os.listdir(checkpoint_dir):
        if not os.path.isfile(os.path.join(checkpoint_dir, f)):
            continue
        if any(p.match(f) for p in patterns):
            matches.append(f)

    return _sorted_nicely(matches)


def get_all_checkpoints(
    checkpoint_dir: str,
    exp_name: str,
    just_files: bool = False,
    device: str = "cpu",
) -> Tuple[Union[List[str], Dict[int, dict]], Optional[dict]]:
    """Load (or list) all checkpoints and statistics for an experiment.

    Args:
        checkpoint_dir: Directory containing checkpoint .pth files.
        exp_name: Experiment name used when saving.
        just_files: If True, return file paths instead of loaded state dicts.
        device: Device to map tensors onto when loading.

    Returns:
        (checkpoints, statistics) where:
          - checkpoints is either a list of file paths (just_files=True)
            or a dict {epoch: state_dict}.
          - statistics is the loaded metrics dict, or None if the
            metrics file doesn't exist.
    """
    model_files = get_model_files(checkpoint_dir, exp_name)
    metadata = {f: extract_metrics(f, exp_name) for f in model_files}

    # Load statistics if available
    stats_path = os.path.join(checkpoint_dir, f"{exp_name}_metrics.pth")
    statistics = torch.load(stats_path, map_location=device) if os.path.exists(stats_path) else None

    if just_files:
        return [os.path.join(checkpoint_dir, f) for f in model_files], statistics

    checkpoints = {}
    for f, meta in metadata.items():
        if meta is not None:
            state = torch.load(os.path.join(checkpoint_dir, f), map_location=device)
            checkpoints[meta["epoch"]] = state

    return checkpoints, statistics
