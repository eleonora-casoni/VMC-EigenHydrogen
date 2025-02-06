import numpy as np 

def trial_wavefunction(x, alpha):
    """Computes the trial wavefunction for given positions x and parameter alpha."""
    result = np.exp(-alpha * x)
    if np.any(x <= 0): 
     raise ValueError("Negative and 0 values for the distance are not allowed for this physical system.")
    
    return result 

def local_energy_func(x, alpha):
    """Computes the local energy for given positions x and parameter alpha."""
    if np.any(x == 0):  # Avoid division by zero
        raise ValueError("Position x must be nonzero to avoid division by zero.")
    return (-1 + alpha * x - (alpha**2) * x / 2) / x
