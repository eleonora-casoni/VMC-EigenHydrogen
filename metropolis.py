import numpy as np
from tqdm import tqdm
from wavefunction import trial_wavefunction
from local_energy import local_energy_func
from alpha_optimization import alpha_opt_on_fly




def metropolis(equilibration_steps, numsteps, numwalkers, alpha):
    position_vec = np.random.uniform(low=2, high=3, size=numwalkers)  # Initialize positions
    initial_pos = position_vec
    alpha_buffer = np.empty(numsteps)
    dE_da_buffer = np.empty(numsteps)
    E_buffer = np.empty(numsteps)

    for j in tqdm(range(numsteps)):  # Progress bar
        for i in range(equilibration_steps):
            new_position_vec = position_vec + 0.1*np.random.randn(numwalkers)
            # Reject moves where x < 0 (those walkers stay at their previous positions)
            valid_moves = new_position_vec > 0
            new_position_vec = np.where(valid_moves, new_position_vec, position_vec)
            p = trial_wavefunction(new_position_vec, alpha) / trial_wavefunction(position_vec, alpha)
            rand_unif_array = np.random.uniform(size=len(p))
            position_vec = np.where(p > rand_unif_array, new_position_vec, position_vec)

        # Optimize α
        alpha, dE_da = alpha_opt_on_fly(position_vec, alpha)
        alpha_buffer[j] = alpha
        dE_da_buffer[j] = dE_da
        E_buffer[j] = np.mean(local_energy_func(position_vec, alpha))

    return position_vec, alpha, alpha_buffer, E_buffer, dE_da_buffer, initial_pos
