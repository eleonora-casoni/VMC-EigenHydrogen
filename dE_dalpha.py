import numpy as np
from local_energy import local_energy_func

def dE_dalpha(x, alpha):
    El = local_energy_func(x, alpha)
    ln_wf = -x  # d(log Ψ)/dα
    return 2 * (np.mean(El * ln_wf) - np.mean(El) * np.mean(ln_wf))
