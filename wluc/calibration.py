
from abc import abstractmethod, ABC
from typing import Tuple, Dict, Self

import numpy as np

from scipy import stats


class ModularConformalCalibration(ABC):

    def __init__(self):
        self.xp = None
        self.fp = None

    @abstractmethod
    def calibration_score(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def interpolation(self, candidate_scores) -> np.ndarray:
        if self.xp is None or self.fp is None:
            raise ValueError(f"this calibration method has not been fitted with `.fit`.")
        return np.interp(candidate_scores, xp=self.xp, fp=self.fp)

    def fit(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        y_true: np.ndarray,
    ) -> Self:
        scores = self.calibration_score(mu=mu, sigma=sigma, y_true=y_true)
        ecdf = stats.ecdf(scores)

        self.xp = ecdf.cdf.quantiles
        self.fp = ecdf.cdf.probabilities

        return self

    def predict(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        y_cand: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert len(mu) == len(sigma)

        mu_ = np.zeros_like(mu)
        sigma_ = np.zeros_like(sigma)

        midpoints = (y_cand[1:] + y_cand[:-1]) / 2.
        midpoints_squared = midpoints*midpoints

        for ix, (m, s) in enumerate(zip(mu, sigma)):
            cdf = self.interpolation(self.calibration_score(mu=m, sigma=s, y_true=y_cand))
            dF = np.diff(cdf)

            mu = np.sum(midpoints*dF)
            m2 = np.sum(midpoints_squared*dF)
            var = m2 - mu*mu

            mu_[ix] = mu
            sigma_[ix] = np.sqrt(max(var, 0.))

        return mu_, sigma_



class ZScoreMCC(ModularConformalCalibration):

    def __init__(self):
        super().__init__()

    def calibration_score(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        return (y_true - mu) / sigma
