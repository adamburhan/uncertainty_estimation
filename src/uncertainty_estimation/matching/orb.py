from typing import Tuple

import numpy as np
import torch
import cv2 as cv


def ORB(
    images: torch.Tensor,  # B, 2, C, H, W  — dim 1: 0=left, 1=right
    device: torch.device,
    max_keypoints: int = 2000,
    max_hamming_distance: int = 64,
    max_epipolar_error: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ORB stereo matching for a batch of rectified stereo pairs.

    Filters:
      - Lowe's ratio test (descriptor quality)
      - Hamming distance <= max_hamming_distance (descriptor quality)
      - |y_left - y_right| <= max_epipolar_error (rectified epipolar constraint)
      - u_left > u_right (positive disparity)

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

        # Lowe's ratio test + stereo geometry filters
        lkps_list = []
        rkps_list = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= 0.75 * n.distance:
                continue
            if m.distance > max_hamming_distance:
                continue
            pt_l = kp1[m.queryIdx].pt
            pt_r = kp2[m.trainIdx].pt
            if abs(pt_l[1] - pt_r[1]) > max_epipolar_error:
                continue
            if pt_l[0] <= pt_r[0]:
                continue
            lkps_list.append(pt_l)
            rkps_list.append(pt_r)

        if not lkps_list:
            _empty()
            continue

        lkps = np.array(lkps_list, dtype=np.float32)  # (P, 2)
        rkps = np.array(rkps_list, dtype=np.float32)  # (P, 2)

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