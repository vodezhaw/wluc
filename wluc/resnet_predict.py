
from pathlib import Path
import json

import numpy as np

import torch
from torch.utils.data import DataLoader

import lightning as pl

from wluc.dataset import load_raw, NoNoiseDataset, StaticNoiseDataset, ScaleInfo
from wluc.resnet import new_resnet, LightningResNet
from wluc.calibration import TorchRescaleCalibration


def pareto_frontier(
    metrics,
    first,
    second,
    first_increasing: bool = False,
    second_increasing: bool = False,
):

    points = np.array([
        [m[first], m[second]]
        for m in metrics
    ])

    # algo is built for minimization in both dimensions by default
    if first_increasing:
        points[:, 0] *= -1.
    if second_increasing:
        points[:, 1] *= -1.

    n = len(metrics)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue

        for j in range(i+1, n):
            if is_dominated[j]:
                continue

            p, q = points[i], points[j]

            if np.all(p <= q) and np.any(p < q):
                is_dominated[j] = True
            elif np.all(q <= p) and np.any(q < p):
                is_dominated[i] = True

    return [
        m
        for m, dom in zip(metrics, is_dominated)
        if not dom
    ]


def main(
    result_folder: Path,
    data_folder: Path,
):
    competition_data = load_raw(data_folder)
    del competition_data['train']  # save some RAM and we don't need training data here
    del competition_data['labels']

    predictions = []

    for fold_folder in result_folder.glob("fold-*"):
        metrics_file = fold_folder / "metrics.jsonl"

        scale_info = ScaleInfo(**torch.load(str(fold_folder / "scale_info.pt")))
        data = torch.load(str(fold_folder / "data.pt"))

        with metrics_file.open('r') as fin:
            metrics_ = [json.loads(line.strip()) for line in fin]

        frontier = pareto_frontier(
            metrics=metrics_,
            first="val_nll",
            second="val_mse",
            first_increasing=False,
            second_increasing=False,
        )

        for m in frontier:
            checkpoint = fold_folder / f"model-{m['num']:02d}.ckpt"

            model = LightningResNet.load_from_checkpoint(
                checkpoint,
                model=new_resnet(),
                epochs=30,
                fold_dir=fold_folder,
            )

            trainer = pl.Trainer(
                accelerator="auto",
                devices="auto",
            )

            calib_data = StaticNoiseDataset(
                samples=data['x_calib'],
                labels=data['y_calib'],
                mask=competition_data['mask'],
                noise_sigma=competition_data['challenge_params']['noise_sigma'],
                scale_info=scale_info,
            )
            calib_predictions = trainer.predict(model, DataLoader(calib_data, batch_size=256, shuffle=False))
            mu_calib = torch.cat([b['mu'] for b in calib_predictions], dim=0)
            sigma_calib = torch.cat([b['sigma'] for b in calib_predictions], dim=0)
            mu_calib, sigma_calib = scale_info.unscale(mu_calib, sigma_calib)

            calibration_model = TorchRescaleCalibration().fit(
                mu=mu_calib,
                sigma=sigma_calib,
                y_true=data['y_calib'],
            )

            test_data = NoNoiseDataset(
                samples=competition_data['test'],
                mask=competition_data['mask'],
                scale_info=scale_info,
            )
            test_predictions = trainer.predict(model, DataLoader(test_data, batch_size=256, shuffle=False))
            mu_test = torch.cat([b['mu'] for b in test_predictions], dim=0)
            sigma_test = torch.cat([b['sigma'] for b in test_predictions], dim=0)
            mu_test, sigma_test = scale_info.unscale(mu_test, sigma_test)

            mu_test, sigma_test = calibration_model.predict(mu=mu_test, sigma=sigma_test)

            pred_item = {
                "fold": fold_folder.name,
                "model": checkpoint.stem,
                "mu": mu_test,
                "sigma": sigma_test,
            }
            predictions.append(pred_item)

    torch.save(predictions, str(result_folder / "calibrated_predictions.pt"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-folder", dest="data_folder", type=Path, required=False, default=Path("./data/"))
    parser.add_argument("-o", "--output-folder", dest="output_folder", type=Path, required=False, default=Path("./out/"))
    args = parser.parse_args()

    main(
        result_folder=args.output_folder,
        data_folder=args.data_folder,
    )
