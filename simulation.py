import numpy as np 
from tqdm import tqdm

def trial_wavefunction(x, alpha):
    """Computes the trial wavefunction for given positions x and parameter alpha."""
    if alpha < -1000:  
     raise ValueError("Too large negative alpha causes numerical instability.")

    result = np.where(x > 0, np.exp(-alpha * x), 0)
    return result 

def local_energy_func(x, alpha):
    """Computes the local energy for given positions x and parameter alpha."""
    
    return -1/x - (alpha**2)/2 + alpha/x

def dE_dalpha(x, alpha):
    El = local_energy_func(x, alpha)
    ln_wf = -x  # d(log Ψ)/dα
    return 2 * (np.mean(El * ln_wf) - np.mean(El) * np.mean(ln_wf))

def alpha_opt_on_fly(position_vec, alpha):
    dE_da = dE_dalpha(position_vec, alpha)
    alpha = alpha - 0.01 * dE_da  # Learning rate = 0.01
    return alpha, dE_da


def metropolis(equilibration_steps, numsteps, numwalkers, alpha):
    position_vec = np.random.uniform(low=2, high=3, size=numwalkers)  
    initial_pos = position_vec
    alpha_buffer = np.empty(numsteps)
    dE_da_buffer = np.empty(numsteps)
    E_buffer = np.empty(numsteps)

    for j in tqdm(range(numsteps)):  
        for i in range(equilibration_steps):
            new_position_vec = position_vec + 0.1 * np.random.randn(numwalkers)
            denominator = trial_wavefunction(position_vec, alpha)
            numerator = trial_wavefunction(new_position_vec, alpha)
            invalid_indices = denominator == 0
            if np.any(invalid_indices):
                print("Error: Division by zero detected in denominator during p calculation.")
                raise ValueError("Division by zero detected in p calculation.")
            p = numerator / denominator
            rand_unif_array = np.random.uniform(size=len(p))
            position_vec = np.where(p > rand_unif_array, new_position_vec, position_vec)

        # Optimize α
        alpha, dE_da = alpha_opt_on_fly(position_vec, alpha)
        alpha_buffer[j] = alpha
        dE_da_buffer[j] = dE_da
        E_buffer[j] = np.mean(local_energy_func(position_vec, alpha))

    return position_vec, alpha, alpha_buffer, E_buffer, dE_da_buffer, initial_pos
