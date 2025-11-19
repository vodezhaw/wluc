# Weak Lensing Uncertainty Challenge

## Setup

Developed with `python 3.13`. This repository uses `uv` to manage python dependencies and environments.
Dependencies are listed in `pyproject.toml`.

Our code expects the challenge `*.npy` data files to be located in `./data/`. Download and copy the challenge data into
that directory before trying to run anything.

## Submission

To reproduce our submission run:

* Training models: `uv run python -m wluc.resnet -d ./data/ -o ./out/`
* Prediction and Calibration: `uv run python -m wluc.resnet_predict -d ./data/ -o ./out/`
* Ensemble and create submission: `uv run python -m wluc.ensemble_predictions -i ./out/calibrated_predictions.pt -o submission.zip`
