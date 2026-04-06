"""Reproduce the cv2.findFundamentalMat crash from the stereo sweep.

Rebuilds the exact train dataloader used by train_stereo_cov.py (same seed,
same num_workers=0 for determinism) and calls ORB on each batch inside a
try/except. On crash, dumps the offending inputs so they can be probed
offline by probe_findfundamentalmat.py.

Launch (mirrors the failing sweep config):
    python -m scripts.debug.repro_orb_crash \
        dataset=sss dataset.stereo_config=horizontal_5cm training.seed=42
"""

import random
from pathlib import Path

import cv2 as cv
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    # Deferred heavy imports (same pattern as train_stereo_cov.py)
    from uncertainty_estimation.matching.orb import ORB
    from uncertainty_estimation.training.data.semistaticsim import SemiStaticSimStereoDataset
    from uncertainty_estimation.training.data.tartanair import TartanAirLiveDataset

    print(OmegaConf.to_yaml(cfg))

    # Same seeding as training script
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cfg.dataset.name == "semistaticsim":
        train_dataset = SemiStaticSimStereoDataset(cfg.dataset, cfg.augmentation, split="train")
    elif cfg.dataset.name == "tartanair":
        train_dataset = TartanAirLiveDataset(cfg.dataset, cfg.augmentation, split="train")
    else:
        raise ValueError(f"Unknown dataset '{cfg.dataset.name}'")

    # num_workers=0 for deterministic ordering and clean tracebacks.
    # Use a fixed-generator shuffle so iteration order matches the training run
    # as closely as possible (it will still differ slightly from num_workers=4
    # runs, but the failing image set is the same).
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
    )

    device = torch.device("cpu")  # ORB is CPU-only; no GPU needed
    matching_fn = lambda images, K: ORB(
        images, device, K,
        max_keypoints=cfg.matching.max_keypoints,
        max_hamming_distance=cfg.matching.max_hamming,
        lowe_ratio=cfg.matching.lowe_ratio,
        ransac_reproj_threshold=cfg.matching.ransac_reproj_threshold,
    )

    dump_dir = Path("orb_crash_dump")
    dump_dir.mkdir(exist_ok=True)

    print(f"Iterating {len(loader)} batches of size {cfg.training.train_batch_size}...")
    for batch_idx, batch in enumerate(loader):
        images = batch["images"]  # (B, 2, 1, H, W) on CPU
        K = batch["K_inv"]  
        K = torch.linalg.inv(K)  # (B, 3, 3)
        try:
            left_kps, right_kps, masks = matching_fn(images, K)
        except cv.error as e:
            print(f"\n[CRASH] batch {batch_idx}: {e}")
            # Re-run ORB one item at a time to find which sample blew up,
            # and replay the ORB descriptor pipeline to reconstruct the exact
            # point set passed to findFundamentalMat.
            orb = cv.ORB_create(cfg.matching.max_keypoints)
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
            for i in range(images.shape[0]):
                left_img = (images[i, 0, 0] * 255.0).numpy().astype(np.uint8)
                right_img = (images[i, 1, 0] * 255.0).numpy().astype(np.uint8)

                kp1, des1 = orb.detectAndCompute(left_img, None)
                kp2, des2 = orb.detectAndCompute(right_img, None)
                if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                    continue
                matches = bf.knnMatch(des1, des2, k=2)

                lkps_list, rkps_list = [], []
                for pair in matches:
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance >= cfg.matching.lowe_ratio * n.distance:
                        continue
                    if m.distance > cfg.matching.max_hamming:
                        continue
                    lkps_list.append(kp1[m.queryIdx].pt)
                    rkps_list.append(kp2[m.trainIdx].pt)

                if len(lkps_list) < 8:
                    continue

                lkps = np.array(lkps_list, dtype=np.float32)
                rkps = np.array(rkps_list, dtype=np.float32)

                try:
                    cv.findFundamentalMat(
                        lkps, rkps, cv.FM_RANSAC,
                        ransacReprojThreshold=cfg.matching.ransac_reproj_threshold,
                    )
                except cv.error as inner:
                    print(f"  -> offending sample: item {i} in batch {batch_idx}")
                    print(f"     lkps shape: {lkps.shape}, rkps shape: {rkps.shape}")
                    print(f"     lkps dtype: {lkps.dtype}, contiguous: {lkps.flags['C_CONTIGUOUS']}")
                    print(f"     unique lkps: {len(np.unique(lkps, axis=0))}/{len(lkps)}")
                    print(f"     unique rkps: {len(np.unique(rkps, axis=0))}/{len(rkps)}")
                    np.save(dump_dir / "lkps.npy", lkps)
                    np.save(dump_dir / "rkps.npy", rkps)
                    np.save(dump_dir / "left_img.npy", left_img)
                    np.save(dump_dir / "right_img.npy", right_img)
                    with open(dump_dir / "meta.txt", "w") as f:
                        f.write(
                            f"batch_idx={batch_idx} item={i}\n"
                            f"stereo_config={cfg.dataset.stereo_config}\n"
                            f"seed={seed}\n"
                            f"ransac_reproj_threshold={cfg.matching.ransac_reproj_threshold}\n"
                            f"opencv_version={cv.__version__}\n"
                            f"error={inner}\n"
                        )
                    print(f"     dumped to {dump_dir.resolve()}")
                    return
            print("  -> could not reproduce per-item; crash may depend on batch order")
            return

        n_valid_total = int(masks.sum().item())
        print(f"  batch {batch_idx:4d}: ok, {n_valid_total} total matches")

    print("\nNo crash encountered across full epoch.")


if __name__ == "__main__":
    main()
