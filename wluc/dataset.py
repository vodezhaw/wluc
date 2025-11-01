
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset


def load_raw(data_dir = Path("./data/")) -> dict:
    mask_file = data_dir / "WIDE12H_bin2_2arcmin_mask.npy"
    train_file = data_dir / "WIDE12H_bin2_2arcmin_kappa.npy"
    test_file = data_dir / "WIDE12H_bin2_2arcmin_kappa_noisy_test.npy"
    label_file = data_dir / "label.npy"

    mask = torch.from_numpy(np.load(mask_file))

    train = torch.from_numpy(np.load(train_file)).float()

    label = torch.from_numpy(np.load(label_file)).float()

    test = torch.from_numpy(np.load(test_file)).float()

    pixel_per_arcmin = 2
    galaxies_per_arcmin2 = 30
    noise_level = 0.4 / (2*galaxies_per_arcmin2*pixel_per_arcmin*pixel_per_arcmin)**.5
    return {
        "train": train,
        "labels": label,
        "test": test,
        "mask": mask,
        "challenge_params": {
            "pixel_arcmin": pixel_per_arcmin,
            "pixel_radians": pixel_per_arcmin / 60. / 180. * np.pi,
            "galaxies_per_arcmin2": galaxies_per_arcmin2,
            "noise_level": noise_level,
            "n_cosmologies": 101,
            "n_realizations": 256,
            "img_w": 1424,
            "img_h": 176,
            "n_test": 4000,
        }
    }


class NoisingDataset(Dataset):

    def __init__(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        noise_level: float,
    ):
        self.samples = samples
        self.labels = labels
        self.mask = mask
        self.noise_level = noise_level

        self.n_samples = self.samples.shape[0]

        sigma_img, mu_img = torch.std_mean(self.samples, dim=0)
        self.sigma_img = sigma_img
        self.mu_img = mu_img

        self.scaled_noise_level = noise_level / self.sigma_img

        sigma_y, mu_y = torch.std_mean(self.labels, dim=0)
        self.sigma_y = sigma_y
        self.mu_y = mu_y

        self.scaled_labels = self.labels.clone()
        self.scaled_labels = (self.scaled_labels - self.mu_y) / self.sigma_y

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> dict:
        x = self.samples[index].clone()
        z = (x - self.mu_img) / self.sigma_img
        noisy = z + self.scaled_noise_level*torch.randn_like(z)
        out = torch.zeros_like(self.mask, dtype=torch.float32)
        out[self.mask] = noisy
        return {
            "x": out,
            "y": self.scaled_labels[index],
        }
