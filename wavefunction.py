import numpy as np 

def trial_wavefunction(x, alpha):
    """Computes the trial wavefunction for given positions x and parameter alpha."""
    if isinstance(x, np.ndarray):
        return np.exp(-alpha * x) * (x > 0)  # Element-wise multiplication applies the condition
    return np.exp(-alpha * x) if x > 0 else 0

def local_energy_func(x, alpha):
    """Computes the local energy for given positions x and parameter alpha."""
    if np.any(x == 0):  # Avoid division by zero
        raise ValueError("Position x must be nonzero to avoid division by zero.")
    return (-1 + alpha * x - (alpha**2) * x / 2) / x
