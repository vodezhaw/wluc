
from abc import abstractmethod, ABC
from typing import Tuple, Dict, Self

import numpy as np


class ModularConformalCalibration(ABC):

    def __init__(self):
        self.xp = None
        self.fp = None

    @abstractmethod
    def calibration_score(
        self,
        pred: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def fit(
        self,
        pred: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> Self:
        scores = self.calibration_score(pred, y_true)
        xp = np.sort(scores)
        fp = np.arange(len(xp)) / len(xp)

        self.xp = xp
        self.fp = fp

        return self

    def predict(
        self,
        pred: Dict[str, np.ndarray],
        n_samples: int,
        y_min: float,
        y_max: float,
        n_discretization: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.xp is None or self.fp is None:
            raise ValueError(f"this calibration method has not been fitted with `.fit`.")

        y_cand = np.linspace(y_min, y_max, n_discretization)

        ys = np.zeros((n_samples, n_discretization))
        scores = np.zeros((n_samples, n_discretization))

        for ix in range(n_samples):
            ys[ix, :] = y_cand

        for jx in range(n_discretization):
            scores[:, jx] = self.calibration_score(pred, ys[:, jx])

        cdf = np.interp(
            x=scores,
            xp=self.xp,
            fp=self.fp,
        )

        mu = np.zeros(n_samples)
        sigma = np.zeros(n_samples)

        for ix in range(n_samples):
            y = ys[ix, :]
            cdf_ = cdf[ix, :]

            pos = y >= 0.
            neg = y <= 0.

            mean = np.trapezoid(1.0 - cdf_[pos], y[pos]) - np.trapezoid(cdf_[neg], y[neg])

            moment2 = 2.*(np.trapezoid(y[pos]*(1. - cdf_[pos]), y[pos]) - np.trapezoid(y[neg]*cdf_[neg], y[neg]))

            var = moment2 - mean*mean
            var = max(var, 0.)
            std = np.sqrt(var)

            mu[ix] = mean
            sigma[ix] = std

        return mu, sigma


class ZScoreMCC(ModularConformalCalibration):

    def __init__(self):
        super().__init__()

    def calibration_score(
        self,
        pred: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> np.ndarray:
        mu = pred['mu']
        sigma = pred['sigma']
        return (y_true - mu) / sigma
