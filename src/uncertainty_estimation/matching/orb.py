from typing import Tuple
import numpy as np

import torch
import cv2 as cv


def ORB(
    images: torch.Tensor,  # B, 2, C, H, W  — dim 1: 0=left, 1=right
    device: torch.device,
    max_keypoints: int = 500,
    max_hamming_distance: int = 64,
    max_epipolar_error: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ORB stereo matching for a batch of rectified stereo pairs.

    Filters:
      - Hamming distance <= max_hamming_distance (descriptor quality)
      - |y_left - y_right| <= max_epipolar_error (rectified epipolar constraint)
      - u_left > u_right (positive disparity)

    Returns:
        left_kps:  (B, P, 2) left keypoint pixel coords, padded
        right_kps: (B, P, 2) right keypoint pixel coords, padded
        masks:     (B, P)    1 for valid matches, 0 for padding
    """
    orb = cv.ORB_create(max_keypoints)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    left_kps = []
    right_kps = []
    masks = []
    for i in range(images.shape[0]):  # batch entries
        left_img  = (images[i, 0, 0] * 255.0).to("cpu").detach().numpy().astype(np.uint8)
        right_img = (images[i, 1, 0] * 255.0).to("cpu").detach().numpy().astype(np.uint8)

        kp1, des1 = orb.detectAndCompute(left_img,  None)
        kp2, des2 = orb.detectAndCompute(right_img, None)

        if des1 is None or des2 is None:
            left_kps.append(torch.zeros(0, 2))
            right_kps.append(torch.zeros(0, 2))
            masks.append(torch.zeros(0))
            continue

        matches = [m for m in bf.match(des1, des2) if m.distance <= max_hamming_distance]
        if not matches:
            left_kps.append(torch.zeros(0, 2))
            right_kps.append(torch.zeros(0, 2))
            masks.append(torch.zeros(0))
            continue

        lkps = np.array([kp1[m.queryIdx].pt for m in matches])  # (P, 2)
        rkps = np.array([kp2[m.trainIdx].pt for m in matches])  # (P, 2)

        # Rectified epipolar constraint: rows must match
        valid = np.abs(lkps[:, 1] - rkps[:, 1]) <= max_epipolar_error
        # Positive disparity: left x > right x
        valid &= lkps[:, 0] > rkps[:, 0]
        lkps, rkps = lkps[valid], rkps[valid]

        left_kps.append(torch.from_numpy(lkps).float())
        right_kps.append(torch.from_numpy(rkps).float())
        masks.append(torch.ones(len(lkps)))

    left_kps  = torch.nn.utils.rnn.pad_sequence(left_kps,  batch_first=True, padding_value=0.0)
    right_kps = torch.nn.utils.rnn.pad_sequence(right_kps, batch_first=True, padding_value=0.0)
    masks     = torch.nn.utils.rnn.pad_sequence(masks,     batch_first=True, padding_value=0.0)

    return left_kps.to(device), right_kps.to(device), masks.to(device)
