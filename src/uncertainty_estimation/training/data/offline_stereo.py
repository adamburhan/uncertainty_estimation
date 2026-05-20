from torch.utils.data import Dataset
import numpy as np
import torch


class OfflineStereoDataset(Dataset):
    """Patches + scalar reprojection error.

    Drops samples with any invalid depth so image/depth/both modalities
    all train on identical indices. Normalization stats are computed on
    the train split and reused for test via the `stats` argument.
    """

    def __init__(self, path, stats=None):
        data = np.load(path)
        img = data["img_patch"]
        depth = data["depth_patch"]
        target = data["reproj_error"]

        valid = np.all(np.isfinite(depth) & (depth > 0), axis=(1, 2))
        self.img = img[valid]
        self.depth = depth[valid].astype(np.float32)
        self.target = target[valid].astype(np.float32)

        if stats is None:
            img_f = self.img.astype(np.float32) / 255.0
            log_d = np.log(self.depth)
            stats = {
                "img_mean": img_f.reshape(-1, 3).mean(0),
                "img_std":  img_f.reshape(-1, 3).std(0),
                "depth_mean": float(log_d.mean()),
                "depth_std":  float(log_d.std()),
            }
        self.stats = stats
        self._img_mean = torch.tensor(stats["img_mean"], dtype=torch.float32).view(3, 1, 1)
        self._img_std  = torch.tensor(stats["img_std"],  dtype=torch.float32).view(3, 1, 1)
        self._d_mean = float(stats["depth_mean"])
        self._d_std  = float(stats["depth_std"])

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.img[idx]).permute(2, 0, 1).float() / 255.0
        img = (img - self._img_mean) / self._img_std

        depth = torch.from_numpy(np.log(self.depth[idx])).unsqueeze(0).float()
        depth = (depth - self._d_mean) / self._d_std

        target = torch.tensor(np.log1p(self.target[idx]), dtype=torch.float32)
        return img, depth, target
