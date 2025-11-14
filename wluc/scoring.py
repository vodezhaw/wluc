
import torch

def official_scoring(
    true_cosmology,
    inferred_cosmology,
    error_bars,
):
    scale_factor = 1000

    with torch.no_grad():
        err = true_cosmology - inferred_cosmology
        sq_err = err*err

        error_bar2 = error_bars*error_bars

        score = -torch.sum(sq_err / error_bar2 + torch.log(error_bar2) + scale_factor*sq_err, dim=1)
        score = torch.mean(score).item()
    return max(score, -1e6)


def hit_rate(
    true_cosmology,
    inferred_cosmology,
    error_bars,
) -> float:
    hi = inferred_cosmology + 1.96*error_bars
    lo = inferred_cosmology - 1.96*error_bars

    hits = (true_cosmology < hi) & (true_cosmology > lo)

    return (hits.sum() / hits.nelement()).item()


def precision(
    true_cosmology,
    inferred_cosmology,
    error_bars,
) -> float:
    return (error_bars.mean(dim=0).sum()).item()