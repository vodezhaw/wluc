
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from sklearn.model_selection import KFold

from torchvision.models import resnet18

import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from wluc.dataset import load_raw, NoisingDataset, compute_scaling
from wluc.scoring import scoring


def new_resnet(
    out_features: int = 2,
):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=model.conv1.bias is not None,
    )
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=out_features,
        bias=model.fc.bias is not None,
    )
    return model


class LightningResNet(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()
        self.model = model

    def batch_loss(self, batch):
        x, y = batch
        mu = self.model(x.unsqueeze(dim=1))
        return F.l1_loss(mu, y)

    def training_step(self, batch, batch_idx):
        loss = self.batch_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.batch_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        mu = self.model(x.unsqueeze(1))
        return {
            'mu': mu,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-4)
        return optimizer


def main(
        data_folder: Path = Path("./data"),
):
    raw = load_raw(data_folder)

    n_splits = 8

    nuisance_ixs = np.arange(raw['challenge_params']['n_realizations'])
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0xdeadbeef)
    splits = [t for _, t in splitter.split(nuisance_ixs)]

    flat_dim = raw['train'].shape[-1]

    truths = []
    mus = []
    sigmas = []
    for k in range(n_splits):
        test_ix = k % n_splits
        val_ix = (k + 1) % n_splits

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

        lit_model = LightningResNet(
            model=new_resnet(out_features=2),
        )

        check_pointing = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename=f"split-{k}-best.ckpt",
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
        )
        trainer = pl.Trainer(
            max_epochs=50,
            accelerator="auto",
            devices="auto",
            callbacks=[check_pointing, early_stopping],
        )

        trainer.fit(
            lit_model,
            DataLoader(train_dataset, batch_size=256, shuffle=True),
            DataLoader(val_dataset, batch_size=256, shuffle=False),
        )

        best_model_path = check_pointing.best_model_path
        best_model = LightningResNet.load_from_checkpoint(best_model_path)


        batched_preds = trainer.predict(best_model, DataLoader(test_dataset, batch_size=256, shuffle=False))
        mu_ = torch.cat([b['mu'] for b in batched_preds], dim=0)
        sigma_ = torch.ones_like(mu_)

        mu, sigma = scale_info.unscale(mu_, sigma_)

        score = scoring(
            true_cosmology=y_test,
            inferred_cosmology=mu,
            error_bars=sigma,
        )
        print(f"Split: {k} -- Score: {score:.3f}")
        truths.append(y_test)
        mus.append(mu)
        # sigmas.append(sigma)

    torch.save({"y_true": truths, "mu": mus}, "./calibration_data_resnet.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', dest="data", type=Path, required=False, default=Path("./data"))
    args = parser.parse_args()

    main(
        data_folder=args.data,
    )
