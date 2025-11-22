
from pathlib import Path

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import torch

import numpy as np


from wluc.scoring import official_scoring
from wluc.calibration import TorchRescaleCalibration

def coverage(
    y_true: torch.Tensor,
    hi: torch.Tensor,
    lo: torch.Tensor,
):
    hits = (y_true >= lo) & (y_true < hi)
    return (torch.sum(hits, dim=0) / len(y_true))


def main(
    in_file: Path,
):
    data = torch.load(in_file)

    eval_items = []
    for fold_ix, elem in enumerate(data):
        mu = elem['mu_calib']
        sigma = elem['sigma_calib']
        y = elem['y_calib']
        ixs = np.arange(len(mu))

        train_ixs, test_ixs = train_test_split(
            ixs,
            test_size=len(mu) // 2,
            shuffle=True,
            random_state=0xdeadbeef,
        )

        mu_train = mu[torch.from_numpy(train_ixs)]
        mu_test = mu[torch.from_numpy(test_ixs)]
        sigma_train = sigma[torch.from_numpy(train_ixs)]
        sigma_test = sigma[torch.from_numpy(test_ixs)]
        y_train = y[torch.from_numpy(train_ixs)]
        y_test = y[torch.from_numpy(test_ixs)]

        calib = TorchRescaleCalibration().fit(mu=mu_train, sigma=sigma_train, y_true=y_train)
        mu_pred, sigma_pred = calib.predict(mu=mu_test, sigma=sigma_test)

        eval_item = {
            "fold": elem['fold'],
            "epoch": elem['model'][-2:],
            "before": {
                "score": official_scoring(true_cosmology=y_test, inferred_cosmology=mu_test, error_bars=sigma_test),
                "r2": r2_score(y_true=y_test.numpy(), y_pred=mu_test.numpy()),
                "coverage": coverage(y_true=y_test, hi=mu_test + sigma_test, lo=mu_test - sigma_test).mean().item(),
            },
            "after": {
                "score": official_scoring(true_cosmology=y_test, inferred_cosmology=mu_pred, error_bars=sigma_pred),
                "r2": r2_score(y_true=y_test.numpy(), y_pred=mu_pred.numpy()),
                "coverage": coverage(y_true=y_test, hi=mu_pred + sigma_pred, lo=mu_pred - sigma_pred).mean().item(),
            },
            "scale_omega_m": calib.scale_factor[0].item(),
            "scale_s_8": calib.scale_factor[1].item(),
        }

        eval_items.append(eval_item)

    eval_items = sorted(eval_items, key=lambda e: (e['fold'], e['epoch']))

    agg = {
        "before": {
            "score": np.array([item['before']['score'] for item in eval_items]),
            "r2": np.array([item['before']['r2'] for item in eval_items]),
            "coverage": np.array([item['before']['coverage'] for item in eval_items]),
        },
        "after": {
            "score": np.array([item['after']['score'] for item in eval_items]),
            "r2": np.array([item['after']['r2'] for item in eval_items]),
            "coverage": np.array([item['after']['coverage'] for item in eval_items]),
        },
        "scale_omega_m": np.array([item['scale_omega_m'] for item in eval_items]),
        "scale_s_8": np.array([item['scale_s_8'] for item in eval_items]),
    }

    mean = {
        "fold": "fold-100",
        "epoch": "99",
        "before": {
            "score": agg['before']['score'].mean(),
            "r2": agg['before']['r2'].mean(),
            "coverage": agg['before']['coverage'].mean(),
        },
        "after": {
            "score": agg['after']['score'].mean(),
            "r2": agg['after']['r2'].mean(),
            "coverage": agg['after']['coverage'].mean(),
        },
        "scale_omega_m": agg['scale_omega_m'].mean(),
        "scale_s_8": agg['scale_s_8'].mean(),
    }

    std = {
        "fold": "fold-101",
        "epoch": "99",
        "before": {
            "score": agg['before']['score'].std(),
            "r2": agg['before']['r2'].std(),
            "coverage": agg['before']['coverage'].std(),
        },
        "after": {
            "score": agg['after']['score'].std(),
            "r2": agg['after']['r2'].std(),
            "coverage": agg['after']['coverage'].std(),
        },
        "scale_omega_m": agg['scale_omega_m'].std(),
        "scale_s_8": agg['scale_s_8'].std(),
    }

    for item in eval_items + [mean, std]:
        fold = int(item['fold'].split('-')[1]) + 1
        epoch = int(item['epoch'])

        s_om = item['scale_omega_m']
        s_s8 = item['scale_s_8']

        b = item['before']
        a = item['after']

        print(f"{fold} & {epoch} & {s_om:.2f} & {s_s8:.2f} & {b['score']:.2f} & {b['r2']:.3f} & {b['coverage']:.3f} & {a['score']:.2f} & {a['r2']:.3f} & {a['coverage']:.3f} \\\\")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-data", dest="data_file", type=Path, required=True)
    args = parser.parse_args()

    main(
        in_file=args.data_file,
    )
