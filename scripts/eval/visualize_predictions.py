"""Dump per-keypoint predictions for all 10 A-experiment configs to .npz files.

Left-image-anchored: residuals = left_kps - reproject(right_kps, depth_right -> left)
"""

import re
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from uncertainty_estimation.geometry.stereo import extract_covs, reproject
from uncertainty_estimation.models.factory import build_model
from uncertainty_estimation.training.data.semistaticsim import (
    SemiStaticSimStereoDataset,
    stereo_collate,
)
from uncertainty_estimation.training.trainer import _lookup_depth


CKPT_ROOT = Path("/home/mila/a/adam.burhan/scratch/stereo-UQ/checkpoints")
EXPERIMENT = "A_stereo"
LOSS_TAG   = "bearing_nll_real"
SEED       = 0
DEVICE     = "cuda"
OUT_DIR    = Path("outputs/eval")

STEREO_CONFIGS = [
    "horizontal_5cm",  "horizontal_10cm", "horizontal_20cm", "horizontal_50cm", "horizontal_100cm",
    "vertical_5cm",    "vertical_10cm",   "vertical_20cm",   "vertical_50cm",   "vertical_100cm",
]

SAMPLE_IDXS = [0, 5, 10, 15, 20]


def find_best_ckpt(stereo: str, seed: int) -> Path:
    """Pick the best_epoch=*.pth checkpoint with the lowest loss= for one run."""
    run_dir = CKPT_ROOT / f"{EXPERIMENT}__semistaticsim_{stereo}_{LOSS_TAG}__seed{seed}"
    candidates = list(run_dir.glob("*best_epoch=*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best_epoch=*.pth files in {run_dir}")
    pattern = re.compile(r"loss=(-?\d+\.\d+)")
    return min(candidates, key=lambda p: float(pattern.search(p.name).group(1)))


def dump_for_config(stereo: str, ckpt_path: Path, sample_idxs: list[int]) -> None:
    """Load model + dataset once, run forward on each sample idx, dump one npz per sample."""
    base    = OmegaConf.load("configs/base.yaml")
    dataset = OmegaConf.load("configs/dataset/sss.yaml")
    cfg     = OmegaConf.merge(base, {"dataset": dataset})
    cfg.dataset.stereo_config = stereo

    ds = SemiStaticSimStereoDataset(cfg.dataset, cfg.augmentation, "val", cfg.matching)

    model = build_model(cfg.model).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  {stereo:18s}  epoch={ckpt.get('epoch'):3d}  val={ckpt.get('val_loss'):8.3f}")

    for sample_idx in sample_idxs:
        if sample_idx >= len(ds):
            print(f"    sample {sample_idx}: out of range (val size {len(ds)}), skipping")
            continue
        batch = stereo_collate([ds[sample_idx]])

        with torch.no_grad():
            images = batch["images"].to(DEVICE)
            K_inv  = batch["K_inv"].to(DEVICE)
            T_lr   = batch["T_lr"].to(DEVICE)
            T_rl   = torch.linalg.inv(T_lr)
            K      = torch.linalg.inv(K_inv)

            left_kps    = batch["left_kps"].to(DEVICE)
            right_kps   = batch["right_kps"].to(DEVICE)
            mask        = batch["match_mask"].to(DEVICE).bool()
            depth_left  = batch["depth_left"].to(DEVICE)
            depth_right = batch["depth_right"].to(DEVICE)

            cov_preds = model(images)
            left_covs, _ = extract_covs(cov_preds, left_kps, right_kps)

            depth_at_right_kps = _lookup_depth(depth_right, right_kps)
            left_kps_reproj    = reproject(right_kps, depth_at_right_kps, K, T_rl)
            residuals          = left_kps - left_kps_reproj
            depth_at_left_kps  = _lookup_depth(depth_left, left_kps)

        m = mask[0].cpu().numpy()
        npz_path = OUT_DIR / f"A_real_{stereo}_seed{SEED}_sample{sample_idx:02d}.npz"
        np.savez(
            npz_path,
            image_left      = images[0, 0, 0].cpu().numpy(),
            depth_left      = depth_left[0].cpu().numpy(),
            left_kps        = left_kps[0].cpu().numpy()[m],
            left_kps_reproj = left_kps_reproj[0].cpu().numpy()[m],
            left_covs       = left_covs[0].cpu().numpy()[m],
            residuals       = residuals[0].cpu().numpy()[m],
            kp_depth        = depth_at_left_kps[0].cpu().numpy()[m],
            stereo          = stereo,
            experiment      = "A_real",
            seed            = SEED,
            sample_idx      = sample_idx,
            epoch           = int(ckpt.get("epoch", -1)),
        )
        n_kps = int(m.sum())
        mean_r = float(np.linalg.norm(residuals[0].cpu().numpy()[m], axis=1).mean())
        print(f"    sample {sample_idx:2d}: kps={n_kps:4d}  mean||r||={mean_r:5.3f}px  -> {npz_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dumping {len(STEREO_CONFIGS)} configs x {len(SAMPLE_IDXS)} samples (seed {SEED})")
    for stereo in STEREO_CONFIGS:
        ckpt_path = find_best_ckpt(stereo, SEED)
        dump_for_config(stereo, ckpt_path, SAMPLE_IDXS)
    print("Done.")


if __name__ == "__main__":
    main()
