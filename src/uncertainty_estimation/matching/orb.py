from typing import Tuple
import numpy as np

import torch
import cv2 as cv


def ORB(
    images: torch.Tensor,  # B, 2, C, H, W  — dim 1: 0=left, 1=right
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ORB stereo matching for a batch of rectified stereo pairs.

    Always matches left (index 0) against right (index 1).
    Matches with u_left <= u_right are removed (invalid disparity).

    Returns:
        left_kps:  (B, P, 2) left keypoint pixel coords, padded
        right_kps: (B, P, 2) right keypoint pixel coords, padded
        masks:     (B, P)    1 for valid matches, 0 for padding
    """
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    left_kps = []
    right_kps = []
    masks = []
    for i in range(images.shape[0]):  # batch entries
        kp1, des1 = orb.detectAndCompute(
            (images[i, 0][0] * 255.0).to("cpu").detach().numpy().astype(np.uint8), None
        )
        kp2, des2 = orb.detectAndCompute(
            (images[i, 1][0] * 255.0).to("cpu").detach().numpy().astype(np.uint8), None
        )
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

        lkps = np.array([kp1[m.queryIdx].pt for m in matches])  # (P, 2)
        rkps = np.array([kp2[m.trainIdx].pt for m in matches])  # (P, 2)

        # Remove matches with invalid disparity (rectified stereo: u_left > u_right)
        valid = lkps[:, 0] > rkps[:, 0]
        lkps, rkps = lkps[valid], rkps[valid]

        left_kps.append(torch.from_numpy(lkps).float())
        right_kps.append(torch.from_numpy(rkps).float())
        masks.append(torch.ones(len(lkps)))

    left_kps  = torch.nn.utils.rnn.pad_sequence(left_kps,  batch_first=True, padding_value=0.0)
    right_kps = torch.nn.utils.rnn.pad_sequence(right_kps, batch_first=True, padding_value=0.0)
    masks     = torch.nn.utils.rnn.pad_sequence(masks,     batch_first=True, padding_value=0.0)

    return left_kps.to(device), right_kps.to(device), masks.to(device)
