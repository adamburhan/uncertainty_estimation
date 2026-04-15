"""Pool per-match residuals + diagnostic columns across stereo configs.

One .npz per stereo config, parallel arrays keyed by column name. Only
`match_mask` is applied at collection time (padding is not a real match);
`depth_valid` and `in_bounds_*` are stored as boolean columns so the
analysis stage can slice on any combination.

Columns per valid match:
  r_left_x,  r_left_y       residual in LEFT image  (left_kp - reproj(right→left))
  r_right_x, r_right_y      residual in RIGHT image (right_kp - reproj(left→right))
  kp_left_x, kp_left_y      left keypoint pixel coords
  kp_right_x, kp_right_y    right keypoint pixel coords
  depth_left, depth_right   clamped GT depth at the keypoints
  depth_valid               GT depth within (0.1, max_depth) in BOTH views
  in_bounds_left            reproj(right→left) landed inside the left image
  in_bounds_right           reproj(left→right) landed inside the right image
  scene_id                  source scene (str)
  frame_idx                 frame index within that scene
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from uncertainty_estimation.geometry.stereo import reproject
from uncertainty_estimation.training.data.semistaticsim import (
    SemiStaticSimStereoDataset,
    stereo_collate,
)
from uncertainty_estimation.training.trainer import _get_depth, _in_bounds


MAX_SAMPLES      = 2000       # per-config ceiling
TARGET_N_MATCHES = 100_000    # stop early once hit
BATCH_SIZE       = 8
NUM_WORKERS      = 4
DEVICE           = "cuda"
SPLIT            = "train"

ORIENTATIONS = ["horizontal", "vertical"]
BASELINES    = [5, 10, 20, 50, 100]

OUT_DIR = Path("outputs/eval/residuals")


def collect(stereo: str) -> tuple[dict, dict]:
    base    = OmegaConf.load("configs/base.yaml")
    dataset = OmegaConf.load("configs/dataset/sss.yaml")
    cfg     = OmegaConf.merge(base, {"dataset": dataset})
    cfg.dataset.stereo_config = stereo

    ds = SemiStaticSimStereoDataset(cfg.dataset, cfg.augmentation, SPLIT, cfg.matching)
    n_take = min(MAX_SAMPLES, len(ds))
    subset = torch.utils.data.Subset(ds, list(range(n_take)))
    loader = DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=stereo_collate, pin_memory=False,
    )

    cols: dict[str, list] = {
        "r_left_x": [], "r_left_y": [],
        "r_right_x": [], "r_right_y": [],
        "kp_left_x": [], "kp_left_y": [],
        "kp_right_x": [], "kp_right_y": [],
        "depth_left": [], "depth_right": [],
        "depth_valid": [],
        "in_bounds_left": [], "in_bounds_right": [],
        "scene_id": [], "frame_idx": [],
    }

    n_matches_total = 0
    n_matches_kept = 0
    n_samples_used = 0

    with torch.no_grad():
        for batch in loader:
            B = batch["images"].shape[0]
            n_samples_used += B

            images = batch["images"].to(DEVICE)
            K_inv  = batch["K_inv"].to(DEVICE)
            T_lr   = batch["T_lr"].to(DEVICE)
            T_rl   = torch.linalg.inv(T_lr)
            K      = torch.linalg.inv(K_inv)
            baseline = batch["baseline"].to(DEVICE)
            focal    = K[:, 0, 0]

            left_kps  = batch["left_kps"].to(DEVICE)
            right_kps = batch["right_kps"].to(DEVICE)
            match_mask = batch["match_mask"].to(DEVICE).bool()  # (B, P)

            depth_l, depth_r, depth_valid = _get_depth(
                "gt", batch, left_kps, right_kps, focal, baseline,
                DEVICE, cfg.dataset.max_depth,
            )

            left_reproj  = reproject(right_kps, depth_r, K, T_rl)
            right_reproj = reproject(left_kps,  depth_l, K, T_lr)
            H, W = images.shape[-2], images.shape[-1]
            in_bounds_left  = _in_bounds(left_reproj,  H, W)
            in_bounds_right = _in_bounds(right_reproj, H, W)

            r_left  = left_kps  - left_reproj
            r_right = right_kps - right_reproj

            # Only padding is filtered at collection time.
            valid = match_mask
            n_matches_total += int(valid.sum())

            b_idx, _ = torch.nonzero(valid, as_tuple=True)

            def flat(t):
                return t[valid].cpu().numpy()

            cols["r_left_x"].append(flat(r_left[..., 0]))
            cols["r_left_y"].append(flat(r_left[..., 1]))
            cols["r_right_x"].append(flat(r_right[..., 0]))
            cols["r_right_y"].append(flat(r_right[..., 1]))
            cols["kp_left_x"].append(flat(left_kps[..., 0]))
            cols["kp_left_y"].append(flat(left_kps[..., 1]))
            cols["kp_right_x"].append(flat(right_kps[..., 0]))
            cols["kp_right_y"].append(flat(right_kps[..., 1]))
            cols["depth_left"].append(flat(depth_l))
            cols["depth_right"].append(flat(depth_r))
            cols["depth_valid"].append(flat(depth_valid).astype(bool))
            cols["in_bounds_left"].append(flat(in_bounds_left).astype(bool))
            cols["in_bounds_right"].append(flat(in_bounds_right).astype(bool))

            scene_ids  = np.array(batch["scene_id"])
            frame_idxs = batch["frame_idx"].numpy()
            b_idx_np   = b_idx.cpu().numpy()
            cols["scene_id"].append(scene_ids[b_idx_np])
            cols["frame_idx"].append(frame_idxs[b_idx_np])

            n_matches_kept = sum(len(c) for c in cols["r_left_x"])
            if n_matches_kept >= TARGET_N_MATCHES:
                break

    arrays = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    stats = {
        "n_samples_used": n_samples_used,
        "n_matches_total": n_matches_total,
        "n_matches_kept": n_matches_kept,
    }
    return arrays, stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for orient in ORIENTATIONS:
        for b in BASELINES:
            stereo = f"{orient}_{b}cm"
            print(f"[{stereo}] collecting ...", flush=True)
            arrays, stats = collect(stereo)
            out = OUT_DIR / f"{stereo}.npz"
            np.savez(out, **arrays, **{f"_stat_{k}": v for k, v in stats.items()})
            n = len(arrays["r_left_x"])
            valid_frac = (
                (arrays["depth_valid"] & arrays["in_bounds_left"] & arrays["in_bounds_right"]).mean()
                if n else 0.0
            )
            print(f"  -> {out}  N={n}  samples={stats['n_samples_used']}  "
                  f"valid_frac={valid_frac:.3f}")


if __name__ == "__main__":
    main()
