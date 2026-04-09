"""Dump per-keypoint predictions for the A/B/D experiments to .npz files.

Left-image-anchored: residuals = left_kps - reproject(right_kps -> left).

Per experiment, the correspondence override (real / synthetic / synthetic_3d)
mirrors what trainer._forward_step does, so the residuals dumped here match
the loss residuals each model was trained on.
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
from uncertainty_estimation.training.trainer import _lookup_depth, _project_perturbed_3d


CKPT_ROOT = Path("/home/mila/a/adam.burhan/scratch/stereo-UQ/checkpoints")
SEED      = 0
DEVICE    = "cuda"
OUT_ROOT  = Path("outputs/eval")

ALL_STEREO = [
    "horizontal_5cm",  "horizontal_10cm", "horizontal_20cm", "horizontal_50cm", "horizontal_100cm",
    "vertical_5cm",    "vertical_10cm",   "vertical_20cm",   "vertical_50cm",   "vertical_100cm",
]
SUBSET_STEREO = ["horizontal_5cm", "horizontal_100cm", "vertical_5cm", "vertical_100cm"]

# Each experiment carries everything needed to (a) find its checkpoints and
# (b) reproduce the correspondence override its training loop applied.
EXPERIMENTS = [
    {
        "label":     "A_stereo",
        "loss_tag":  "bearing_nll_real",
        "mode":      "real",
        "sigma":     None,
        "configs":   ALL_STEREO,
        "prefix":    "A_real",
        "out_subdir": "exp_A",
    },
    {
        "label":     "B_falsif",
        "loss_tag":  "bearing_nll_synthetic",
        "mode":      "synthetic",
        "sigma":     2.0,                 # pixels — must match launch script
        "configs":   SUBSET_STEREO,
        "prefix":    "B_synth",
        "out_subdir": "exp_B",
    },
    {
        "label":     "D_3dctrl",
        "loss_tag":  "bearing_nll_synthetic_3d",
        "mode":      "synthetic_3d",
        "sigma":     0.05,                # metres — must match launch script
        "configs":   SUBSET_STEREO,
        "prefix":    "D_synth3d",
        "out_subdir": "exp_D",
    },
]

SAMPLE_IDXS = [0, 10, 15, 20]


def find_best_ckpt(label: str, loss_tag: str, stereo: str, seed: int) -> Path:
    """Pick the best_epoch=*.pth checkpoint with the lowest loss= for one run."""
    run_dir = CKPT_ROOT / f"{label}__semistaticsim_{stereo}_{loss_tag}__seed{seed}"
    candidates = list(run_dir.glob("*best_epoch=*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best_epoch=*.pth files in {run_dir}")
    pattern = re.compile(r"loss=(-?\d+\.\d+)")
    return min(candidates, key=lambda p: float(pattern.search(p.name).group(1)))


def dump_for_config(exp: dict, stereo: str, ckpt_path: Path, sample_idxs: list[int]) -> None:
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

    out_dir = OUT_ROOT / exp["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in sample_idxs:
        if sample_idx >= len(ds):
            print(f"    sample {sample_idx}: out of range (val size {len(ds)}), skipping")
            continue
        batch = stereo_collate([ds[sample_idx]])

        # Deterministic synthetic noise per (config, sample) so reruns reproduce.
        torch.manual_seed(SEED * 100003 + sample_idx)

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

            # GT-based reprojections — same in all modes (these are the "truth" target)
            depth_at_left_kps_orig  = _lookup_depth(depth_left,  left_kps)
            depth_at_right_kps_orig = _lookup_depth(depth_right, right_kps)
            right_kps_reproj = reproject(left_kps,  depth_at_left_kps_orig,  K, T_lr)
            left_kps_reproj  = reproject(right_kps, depth_at_right_kps_orig, K, T_rl)

            # Replicate trainer's per-mode correspondence override.
            mode  = exp["mode"]
            sigma = exp["sigma"]
            if mode == "synthetic":
                # 2D-iso pixel noise: residual = injected noise.
                right_kps = right_kps_reproj + torch.randn_like(right_kps_reproj) * sigma
                left_kps  = left_kps_reproj  + torch.randn_like(left_kps_reproj)  * sigma
                # Matcher mask is no longer meaningful — keep all kps.
                mask = torch.ones_like(mask, dtype=torch.bool)
            elif mode == "synthetic_3d":
                # 3D-iso noise pushed through projection. Need ORIGINAL right_kps
                # for the rl direction, so save before overwriting right_kps.
                right_kps_orig = right_kps
                right_kps, valid_lr = _project_perturbed_3d(
                    left_kps,       depth_at_left_kps_orig,  K, K_inv, T_lr, sigma
                )
                left_kps,  valid_rl = _project_perturbed_3d(
                    right_kps_orig, depth_at_right_kps_orig, K, K_inv, T_rl, sigma
                )
                mask = valid_lr & valid_rl
            elif mode != "real":
                raise ValueError(f"Unknown mode: {mode}")

            # Sample covariance at the (possibly overridden) observation locations.
            left_covs, _ = extract_covs(model(images), left_kps, right_kps)

            residuals         = left_kps - left_kps_reproj
            depth_at_left_kps = _lookup_depth(depth_left, left_kps)

        m = mask[0].cpu().numpy()
        npz_path = out_dir / f"{exp['prefix']}_{stereo}_seed{SEED}_sample{sample_idx:02d}.npz"
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
            experiment      = exp["prefix"],
            mode            = mode,
            sigma           = -1.0 if sigma is None else float(sigma),
            seed            = SEED,
            sample_idx      = sample_idx,
            epoch           = int(ckpt.get("epoch", -1)),
        )
        n_kps = int(m.sum())
        mean_r = float(np.linalg.norm(residuals[0].cpu().numpy()[m], axis=1).mean())
        print(f"    sample {sample_idx:2d}: kps={n_kps:4d}  mean||r||={mean_r:6.3f}px  -> {npz_path.name}")


def main():
    for exp in EXPERIMENTS:
        print(f"=== {exp['label']}  ({exp['mode']}, sigma={exp['sigma']})  "
              f"{len(exp['configs'])} configs x {len(SAMPLE_IDXS)} samples ===")
        for stereo in exp["configs"]:
            ckpt_path = find_best_ckpt(exp["label"], exp["loss_tag"], stereo, SEED)
            dump_for_config(exp, stereo, ckpt_path, SAMPLE_IDXS)
    print("Done.")


if __name__ == "__main__":
    main()
