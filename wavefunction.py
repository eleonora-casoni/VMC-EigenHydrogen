import numpy as np 

def trial_wavefunction(x, alpha):
    """Computes the trial wavefunction for given positions x and parameter alpha."""
    result = np.exp(-alpha * x)
    if np.any(x <= 0): 
     raise ValueError("Negative and 0 values for the distance are not allowed for this physical system.")
    
    return result 

