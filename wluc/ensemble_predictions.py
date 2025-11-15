
from pathlib import Path
from typing import Tuple

import torch

from wluc.dataset import create_submission_file


def ensemble_predictions(
    mus: torch.Tensor,
    sigmas: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    mu_ensemble = torch.mean(mus, dim=0)

    variances = sigmas*sigmas

    var_aleatoric = torch.mean(variances, dim=0)

    var_epistemic = torch.var(mus, dim=0, unbiased=False)

    var_ensemble = var_aleatoric + var_epistemic

    return mu_ensemble, torch.sqrt(var_ensemble.clamp_min(1e-6))



def main(
    data_file: Path,
    out_file: Path,
):
    data = torch.load(data_file)

    mus = torch.stack([d['mu'] for d in data], dim=0)
    sigmas = torch.stack([d['sigma'] for d in data], dim=0)

    mu_ensemble, sigma_ensemble = ensemble_predictions(
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
    args = parser.parse_args()

    main(
        data_file=args.data_file,
        out_file=args.out_file,
    )
