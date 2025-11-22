
from pathlib import Path
from typing import Tuple

import torch

from wluc.dataset import create_submission_file


def inverse_variance_weighing(
    mus: torch.Tensor,
    sigmas: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    variances = sigmas * sigmas

    weights = 1. / variances

    var_out = 1. / torch.sum(weights, dim=0)

    mu_out = torch.sum(mus*weights, dim=0) * var_out

    return mu_out, torch.sqrt(var_out)


def ensemble_predictions(
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    identical_to_submission: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    mu_ensemble = torch.mean(mus, dim=0)

    variances = sigmas*sigmas

    var_aleatoric = torch.mean(variances, dim=0)

    if identical_to_submission:
        diffs = mus - mu_ensemble.unsqueeze(0)
        var_epistemic = torch.mean(diffs*diffs, dim=0)
    else:
        var_epistemic = torch.var(mus, dim=0, unbiased=False)

    var_ensemble = var_aleatoric + var_epistemic

    return mu_ensemble, torch.sqrt(var_ensemble.clamp_min(1e-6))



def main(
    data_file: Path,
    out_file: Path,
    identical_to_submission: bool = False,
):
    data = torch.load(data_file)

    mus = torch.stack([d['mu'] for d in data], dim=0)
    sigmas = torch.stack([d['sigma'] for d in data], dim=0)

    if identical_to_submission:
        mu_ensemble, sigma_ensemble = ensemble_predictions(
            mus=mus,
            sigmas=sigmas,
            identical_to_submission=identical_to_submission,
        )
    else:
        mu_ensemble, sigma_ensemble = inverse_variance_weighing(
            mus=mus,
            sigmas=sigmas,
        )

    create_submission_file(
        mu=mu_ensemble,
        sigma=sigma_ensemble,
        path=out_file,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-data", dest="data_file", type=Path, required=True)
    parser.add_argument('-o', "--out-file", dest="out_file", type=Path, required=True)
    parser.add_argument("--identical-to-submission", dest="identical_to_submission", action="store_true")
    args = parser.parse_args()

    main(
        data_file=args.data_file,
        out_file=args.out_file,
        identical_to_submission=args.identical_to_submission,
    )
