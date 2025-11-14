
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

from wluc.dataset import load_raw, NoisingDataset, compute_scaling, StaticNoiseDataset, NoNoiseDataset
from wluc.scoring import official_scoring, hit_rate, precision


def new_resnet(
    out_features: int = 4,
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
    # model.fc = nn.Linear(
    #     in_features=model.fc.in_features,
    #     out_features=out_features,
    #     bias=model.fc.bias is not None,
    # )
    model.fc = nn.Sequential(
        nn.Linear(in_features=model.fc.in_features, out_features=256, bias=True),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(in_features=256, out_features=out_features, bias=True)
    )
    return model


def r2(y_true, y_pred):
    res = y_true - y_pred
    ss_res = torch.sum(res*res, dim=0)

    y_mean = torch.mean(y_true, dim=0, keepdim=True)
    baseline_res = y_true - y_mean
    ss_tot = torch.sum(baseline_res*baseline_res, dim=0)

    return 1.0 - (ss_res / ss_tot)



class LightningResNet(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
    ):
        super().__init__()
        self.model = model
        self.epochs = epochs

    def forward(self, x):
        out = self.model(x)
        mu = out[:, :2]
        log_var = out[:, 2:]
        log_var = torch.clamp(log_var, min=-6., max=4.)
        return mu, log_var

    def batch_loss(self, mu, log_var, y_true):
        res = y_true - mu
        inv_var = torch.exp(-log_var)
        nll = .5 * (res*res*inv_var + log_var)
        return nll.mean(dim=0).sum()

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, log_var = self(x.unsqueeze(1))
        loss = self.batch_loss(mu, log_var, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_r2", r2(y, mu).mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_mse", F.mse_loss(y, mu), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, log_var = self(x.unsqueeze(1))
        loss = self.batch_loss(mu, log_var, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_r2", r2(y, mu).mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_mse", F.mse_loss(y, mu), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        mu, log_var = self(x.unsqueeze(1))
        sigma = torch.exp(0.5*log_var)
        return {
            'mu': mu,
            'sigma': sigma,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)

        warmup = 5
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=.1,
            total_iters=warmup,
        )
        cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.epochs - warmup,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_annealing],
            milestones=[warmup],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


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
        val_dataset = StaticNoiseDataset(
            samples=x_val,
            labels=y_val,
            mask=raw['mask'],
            noise_sigma=raw['challenge_params']['noise_sigma'],
            scale_info=scale_info,
        )
        test_dataset = StaticNoiseDataset(
            samples=x_test,
            labels=y_test,
            mask=raw['mask'],
            noise_sigma=raw['challenge_params']['noise_sigma'],
            scale_info=scale_info,
        )

        max_epochs = 50
        lit_model = LightningResNet(
            model=new_resnet(),
            epochs=max_epochs,
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
            patience=10,
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            callbacks=[check_pointing, early_stopping],
            gradient_clip_val=1.0,
        )

        trainer.fit(
            lit_model,
            DataLoader(train_dataset, batch_size=256, shuffle=True),
            DataLoader(val_dataset, batch_size=256, shuffle=False),
        )

        best_model_path = check_pointing.best_model_path
        best_model = LightningResNet.load_from_checkpoint(best_model_path, model=new_resnet(), epochs=max_epochs)


        batched_preds = trainer.predict(best_model, DataLoader(test_dataset, batch_size=256, shuffle=False))
        mu_ = torch.cat([b['mu'] for b in batched_preds], dim=0)
        sigma_ = torch.cat([b['sigma'] for b in batched_preds], dim=0)

        mu, sigma = scale_info.unscale(mu_, sigma_)

        score = official_scoring(
            true_cosmology=y_test,
            inferred_cosmology=mu,
            error_bars=sigma,
        )
        hits = hit_rate(
            true_cosmology=y_test,
            inferred_cosmology=mu,
            error_bars=sigma,
        )
        prec = precision(
            true_cosmology=y_test,
            inferred_cosmology=mu,
            error_bars=sigma,
        )
        print(f"Split: {k} -- L1: {F.l1_loss(mu, y_test).item():.3f} -- L2: {F.mse_loss(mu, y_test).item():.3f} == R2: {r2(y_test, mu).mean().item():.3f}")
        print(f"Split {k} -- Score: {score:.3f} -- Hit: {hits:.3f} -- P: {prec:.3f}")
        truths.append(y_test)
        mus.append(mu)
        sigmas.append(sigma)

    torch.save({"y_true": truths, "mu": mus, 'sigma': sigmas}, "./calibration_data_resnet.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', dest="data", type=Path, required=False, default=Path("./data"))
    args = parser.parse_args()

    main(
        data_folder=args.data,
    )
