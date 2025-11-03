
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate

import lightning as pl

from sklearn.model_selection import KFold

import numpy as np

from wluc.dataset import load_raw, NoisingDataset, compute_scaling
from wluc.scoring import scoring


# taken from https://github.com/FAIR-Universe/Cosmology_Challenge/blob/master/Phase_1_Startingkit_WL_CNN_Direct.ipynb
class SimpleCNN(nn.Module):
    def __init__(self, height, width, num_targets):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self._feature_size = self._get_conv_output_size(height, width)

        # Fully connected layers (regressor head)
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_targets)
        )

    def _get_conv_output_size(self, height, width):
        dummy_input = torch.zeros(1, 1, height, width)
        output = self.conv_stack(dummy_input)
        return output.shape.numel()

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        means = x[:, :2]
        log_sigmas = x[:, 2:]  # Predict log(σ) to ensure positivity
        sigmas = torch.exp(log_sigmas)
        return means, sigmas


def kl_loss(mu_pred, sigma_pred, y_true):
    residuals = mu_pred - y_true
    residuals2 = residuals * residuals

    var_pred = sigma_pred * sigma_pred

    point_loss = torch.sum(residuals2 / var_pred, dim=1)
    var_loss = torch.sum(torch.log(var_pred), dim=1)

    return torch.mean(point_loss + var_loss)


class LightningDirect(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = SimpleCNN(
            height=1424,
            width=176,
            num_targets=4,
        )

    def batch_loss(self, batch):
        x, y = batch

        mu, sigma = self.model(x.unsqueeze(1))

        loss = kl_loss(mu, sigma, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.batch_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.batch_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_ix):
        x, _ = batch
        mu, sigma = self.model(x.unsqueeze(1))
        return {
            "mu": mu,
            "sigma": sigma,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            }
        }


def main(
    data_folder: Path = Path("./data"),
):
    raw = load_raw(data_folder)

    nuisance_ixs = np.arange(raw['challenge_params']['n_realizations'])
    splitter = KFold(n_splits=5, shuffle=True, random_state=0xdeadbeef)
    splits = [t for _, t in splitter.split(nuisance_ixs)]

    flat_dim = raw['train'].shape[-1]

    truths = []
    mus = []
    sigmas = []
    for k in range(5):
        test_ix = k % 5
        val_ix = (k + 1) % 5

        test_mask = torch.zeros(raw['challenge_params']['n_realizations'], dtype=bool)
        val_mask = torch.zeros(raw['challenge_params']['n_realizations'], dtype=bool)
        train_mask = torch.ones(raw['challenge_params']['n_realizations'], dtype=bool)

        test_mask[splits[test_ix]] = True
        train_mask[splits[test_ix]] = False

        val_mask[splits[val_ix]] = True
        train_mask[splits[val_ix]] = False

        x_train = raw['train'][:, train_mask, :].view(-1, flat_dim)
        y_train = raw['labels'][:, train_mask, :2].view(-1, 2)
        x_val = raw['train'][:, val_mask, :].view(-1, flat_dim)
        y_val = raw['labels'][:, val_mask, :2].view(-1, 2)
        x_test = raw['train'][:, test_mask, :].view(-1, flat_dim)
        y_test = raw['labels'][:, test_mask, :2].view(-1, 2)

        scale_info = compute_scaling(
            samples=x_train,
            labels=y_train,
        )

        train_dataset = NoisingDataset(
            samples=x_train,
            labels=y_train,
            mask=raw['mask'],
            noise_sigma=raw['challenge_params']['noise_sigma'],
            scale_info=scale_info,
        )
        val_dataset = NoisingDataset(
            samples=x_val,
            labels=y_val,
            mask=raw['mask'],
            noise_sigma=raw['challenge_params']['noise_sigma'],
            scale_info=scale_info,
        )
        test_dataset = NoisingDataset(
            samples=x_test,
            labels=y_test,
            mask=raw['mask'],
            noise_sigma=raw['challenge_params']['noise_sigma'],
            scale_info=scale_info,
        )

        lit_model = LightningDirect()

        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="auto",
            devices="auto",
        )

        trainer.fit(
            lit_model,
            DataLoader(train_dataset, batch_size=64, shuffle=True),
            DataLoader(val_dataset, batch_size=64, shuffle=False),
        )

        batched_preds = trainer.predict(lit_model, DataLoader(test_dataset, batch_size=64, shuffle=False))
        mu_ = torch.cat([b['mu'] for b in batched_preds], dim=0)
        sigma_ = torch.cat([b['sigma'] for b in batched_preds], dim=0)

        mu, sigma = scale_info.unscale(mu_, sigma_)

        score = scoring(
            true_cosmology=y_test,
            inferred_cosmology=mu,
            error_bars=sigma,
        )
        print(f"Split: {k} -- Score: {score:.3f}")
        truths.append(y_test)
        mus.append(mu)
        sigmas.append(sigma)

    torch.save({"y_true": truths, "mu": mus, "sigma": sigmas}, "./calibration_data.pt")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', dest="data", type=Path, required=False, default=Path("./data"))
    args = parser.parse_args()

    main(
        data_folder=args.data,
    )
