from typing import Tuple

import numpy as np
import torch
import cv2 as cv


def orb_single(
    left: torch.Tensor,   # (1, H, W) float in [0, 1]
    right: torch.Tensor,  # (1, H, W)
    K: torch.Tensor,      # (3, 3)
    max_keypoints: int = 2000,
    max_hamming_distance: int = 64,
    lowe_ratio: float = 0.75,
    ransac_reproj_threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ORB matching for a SINGLE stereo pair on CPU.

    Designed to be called from a DataLoader worker (`__getitem__`) so that ORB
    runs in parallel with GPU work. Returns variable-length tensors; the dataset
    is responsible for collating them via a custom collate_fn.

    Returns:
        left_kps:  (P, 2) float32 — empty (0, 2) if matching failed
        right_kps: (P, 2) float32
    """
    # Create the detector per-call: cv.ORB_create() objects are not safe to
    # share across multiprocessing forks.
    orb = cv.ORB_create(max_keypoints)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    left_img = (left[0].numpy() * 255.0).astype(np.uint8)
    right_img = (right[0].numpy() * 255.0).astype(np.uint8)

    empty = (torch.zeros(0, 2), torch.zeros(0, 2))

    kp1, des1 = orb.detectAndCompute(left_img, None)
    kp2, des2 = orb.detectAndCompute(right_img, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return empty

    matches = bf.knnMatch(des1, des2, k=2)
    if not matches:
        return empty

    lkps_list = []
    rkps_list = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance >= lowe_ratio * n.distance:
            continue
        if m.distance > max_hamming_distance:
            continue
        lkps_list.append(kp1[m.queryIdx].pt)
        rkps_list.append(kp2[m.trainIdx].pt)

    if len(lkps_list) < 8:
        return empty

    lkps = np.array(lkps_list, dtype=np.float32)
    rkps = np.array(rkps_list, dtype=np.float32)

    K_np = np.ascontiguousarray(K.numpy(), dtype=np.float64)
    _, inlier_mask = cv.findEssentialMat(
        lkps, rkps, K_np, cv.RANSAC, threshold=ransac_reproj_threshold,
    )
    if inlier_mask is None or inlier_mask.sum() == 0:
        return empty

    inliers = inlier_mask.ravel().astype(bool)
    return torch.from_numpy(lkps[inliers]), torch.from_numpy(rkps[inliers])


def ORB(
    images: torch.Tensor,  # B, 2, C, H, W  — dim 1: 0=left, 1=right
    device: torch.device,
    K: torch.Tensor,
    max_keypoints: int = 2000,
    max_hamming_distance: int = 64,
    lowe_ratio: float = 0.75,
    ransac_reproj_threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ORB matching for a batch of stereo pairs (any baseline direction).

    Filters:
      - Lowe's ratio test (descriptor quality)
      - Hamming distance <= max_hamming_distance (descriptor quality)
      - RANSAC fundamental matrix estimation (geometric consistency)

    Returns:
        left_kps:  (B, P, 2) left keypoint pixel coords, padded
        right_kps: (B, P, 2) right keypoint pixel coords, padded
        masks:     (B, P)    1 for valid matches, 0 for padding
    """
    orb = cv.ORB_create(max_keypoints)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    left_kps = []
    right_kps = []
    masks = []

    def _empty():
        left_kps.append(torch.zeros(0, 2))
        right_kps.append(torch.zeros(0, 2))
        masks.append(torch.zeros(0))

    for i in range(images.shape[0]):
        # Extract grayscale uint8 images
        left_img = (images[i, 0, 0] * 255.0).cpu().detach().numpy().astype(np.uint8)
        right_img = (images[i, 1, 0] * 255.0).cpu().detach().numpy().astype(np.uint8)

        kp1, des1 = orb.detectAndCompute(left_img, None)
        kp2, des2 = orb.detectAndCompute(right_img, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            _empty()
            continue

        matches = bf.knnMatch(des1, des2, k=2)
        if not matches:
            _empty()
            continue

        # Lowe's ratio test + hamming threshold
        lkps_list = []
        rkps_list = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= lowe_ratio * n.distance:
                continue
            if m.distance > max_hamming_distance:
                continue
            lkps_list.append(kp1[m.queryIdx].pt)
            rkps_list.append(kp2[m.trainIdx].pt)

        if len(lkps_list) < 8:  # need >= 8 points for fundamental matrix
            _empty()
            continue

        lkps = np.array(lkps_list, dtype=np.float32)
        rkps = np.array(rkps_list, dtype=np.float32)

        # RANSAC geometric verification via essential matrix
        K_np = np.ascontiguousarray(K[i].detach().cpu().numpy(), dtype=np.float64)
        _, inlier_mask = cv.findEssentialMat(
            lkps, rkps, K_np, cv.RANSAC, threshold=ransac_reproj_threshold,
        )

        if inlier_mask is None or inlier_mask.sum() == 0:
            _empty()
            continue

        inliers = inlier_mask.ravel().astype(bool)
        lkps = lkps[inliers]
        rkps = rkps[inliers]

        left_kps.append(torch.from_numpy(lkps))
        right_kps.append(torch.from_numpy(rkps))
        masks.append(torch.ones(len(lkps)))

    # Handle fully empty batch
    if all(t.shape[0] == 0 for t in left_kps):
        B = images.shape[0]
        return (
            torch.zeros(B, 0, 2, device=device),
            torch.zeros(B, 0, 2, device=device),
            torch.zeros(B, 0, device=device),
        )

    left_kps = torch.nn.utils.rnn.pad_sequence(left_kps, batch_first=True, padding_value=0.0)
    right_kps = torch.nn.utils.rnn.pad_sequence(right_kps, batch_first=True, padding_value=0.0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0.0)

    return left_kps.to(device), right_kps.to(device), masks.to(device)
