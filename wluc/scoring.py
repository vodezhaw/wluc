
import torch

def scoring(
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