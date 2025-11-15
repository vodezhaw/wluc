
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import zipfile

import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset


@dataclass(frozen=True)
class ScaleInfo:
    mu_img: torch.Tensor
    sigma_img: torch.Tensor
    mu_y: torch.Tensor
    sigma_y: torch.Tensor

    def unscale(self, mu, sigma):
        mu_ = mu * self.sigma_y
        mu_ += self.mu_y

        sigma_ = sigma * self.sigma_y

        return mu_, sigma_

    def save(self, path: str | Path) -> None:
        torch.save(asdict(self), str(path))


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
            "noise_sigma": noise_level,
            "n_cosmologies": 101,
            "n_realizations": 256,
            "img_w": 1424,
            "img_h": 176,
            "n_test": 4000,
        }
    }


def compute_scaling(samples, labels) -> ScaleInfo:
    sigma_img, mu_img = torch.std_mean(samples)
    sigma_y, mu_y = torch.std_mean(labels, dim=0)
    return ScaleInfo(
        mu_img=mu_img,
        sigma_img=sigma_img,
        mu_y=mu_y,
        sigma_y=sigma_y,
    )


class NoisingDataset(Dataset):

    def __init__(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        noise_sigma: float,
        scale_info: ScaleInfo,
    ):
        self.samples = samples
        self.labels = labels
        self.mask = mask
        self.noise_sigma = noise_sigma
        self.scale_info = scale_info

        self.n_samples = self.samples.shape[0]

        self.scaled_noise_sigma = self.noise_sigma / self.scale_info.sigma_img

        self.scaled_labels = self.labels.clone()
        self.scaled_labels = (self.scaled_labels - self.scale_info.mu_y) / self.scale_info.sigma_y

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        x = self.samples[index].clone()
        z = (x - self.scale_info.mu_img) / self.scale_info.sigma_img
        noisy = z + self.scaled_noise_sigma*torch.randn_like(z)
        out = torch.zeros_like(self.mask, dtype=torch.float32)
        out[self.mask] = noisy
        return out, self.scaled_labels[index]


class StaticNoiseDataset(Dataset):

    def __init__(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        noise_sigma: float,
        scale_info: ScaleInfo,
    ):
        self.samples = samples
        self.labels = labels
        self.mask = mask
        self.noise_sigma = noise_sigma
        self.scale_info = scale_info

        self.n_samples = self.samples.shape[0]

        self.noised_samples = self.samples + torch.randn_like(self.samples) * noise_sigma

        self.scaled_labels = self.labels.clone()
        self.scaled_labels = (self.scaled_labels - self.scale_info.mu_y) / self.scale_info.sigma_y

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int):
        x = self.noised_samples[index].clone()
        z = (x - self.scale_info.mu_img) / self.scale_info.sigma_img
        out = torch.zeros_like(self.mask, dtype=torch.float32)
        out[self.mask] = z
        return out, self.scaled_labels[index]


class NoNoiseDataset(Dataset):

    def __init__(
        self,
        samples: torch.Tensor,
        mask: torch.Tensor,
        scale_info: ScaleInfo,
    ):
        self.samples = samples
        self.mask = mask
        self.scale_info = scale_info

        self.n_samples = self.samples.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int):
        x = self.samples[index].clone()
        z = (x - self.scale_info.mu_img) / self.scale_info.sigma_img
        out = torch.zeros_like(self.mask, dtype=torch.float32)
        out[self.mask] = z
        return out, self.scale_info.mu_y  # y not used for test data


def create_submission_file(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    path: str | Path,
) -> None:
    assert mu.shape == (4000, 2)
    assert sigma.shape == (4000, 2)

    out = {
        "means": mu.tolist(),
        "errorbars": sigma.tolist(),
    }

    json_str = json.dumps(out, indent=2)

    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("result.json", json_str)
