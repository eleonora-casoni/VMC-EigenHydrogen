import numpy as np
import warnings
from tqdm import tqdm


def trial_wavefunction(x, alpha):
    """
    Compute the trial wavefunction for a given set of positions and a parameter alpha.

    This function calculates the trial wavefunction for a system with spherical symmetry,
    specifically modeling the radial part of the hydrogen atom wavefunction. Since the
    wavefunction is defined only for positive radial distances, negative positions are
    assigned a value of zero to maintain physical consistency. A Gaussian form is used
    for the wavefunction.

    Parameters
    ----------
    x : numpy.ndarray
        Array of position values (one for each walker). Must be non-negative.
    alpha : float
        Variational parameter controlling the wavefunction's decay. Must be a real number.

    Returns
    -------
    numpy.ndarray
        Array of wavefunction values corresponding to the input positions.

    Raises
    ------
    TypeError
        If `alpha` is not a real number (integer or float).
    ValueError
        If `alpha` is less than -1000 or greater than 200, as such values
        lead to numerical instability.
    """
    if not isinstance(alpha, (int, float)):
        raise TypeError(
            "Alpha must be a real number."
        )

    if alpha < -1000:
        raise ValueError(
            "Too large negative alpha causes numerical instability."
        )

    if alpha > 200:
        raise ValueError(
            "Too large alpha causes numerical instability."
        )
    
    result = np.where(x > 0, np.exp(-alpha * x), 0)
    return result


def local_energy_func(x, alpha):
    """
    Compute the local energy for a given set of positions and a variational parameter alpha.

    This function calculates the local energy of a quantum system at different positions,
    estimating the energy of a trial wavefunction.

    Parameters
    ----------
    x : numpy.ndarray
        Array of position values (one for each walker). Must be non-negative.
    alpha : float
        Variational parameter affecting the energy calculation. Must be a real number.

    Returns
    -------
    numpy.ndarray
        Array of local energy values corresponding to the input positions.

    """
    return -1 / x - (alpha**2) / 2 + alpha / x


def dE_dalpha(x, alpha):
    """
    Compute the derivative of the energy with respect to alpha for variational optimization.

    This function calculates the gradient of the energy with respect to the variational
    parameter alpha, which is used in the stochastic gradient descent method to optimize
    alpha 'on the fly'. The derivative is computed as the difference between the mean of
    the product and the product of the means, a technique that reduces noise and stabilizes
    the optimization process.

    Parameters
    ----------
    x : numpy.ndarray
        Array of walker positions after equilibration.
    alpha : float
        Variational parameter influencing the wavefunction.

    Returns
    -------
    float
        The derivative of the energy with respect to alpha.

    Notes
    -----
    - The calculation follows the relation:
      dE/dα = 2 * (⟨E_L * d(log Ψ)/dα⟩ - ⟨E_L⟩ * ⟨d(log Ψ)/dα⟩)
    - The term `ln_wf = -x` represents the derivative of the logarithm of the trial wavefunction Ψ
      with respect to α:
      d(log Ψ)/dα = -x
    - Using this formulation helps reduce noise and improves numerical stability in the optimization process.
    """
    El = local_energy_func(x, alpha)
    ln_wf = -x
    return 2 * (np.mean(El * ln_wf) - np.mean(El) * np.mean(ln_wf))


def alpha_opt_on_fly(position_vec, alpha, learning_rate):
    """
    Perform an on-the-fly update of the variational parameter alpha using gradient descent.

    This function updates alpha using the gradient of the local energy with respect to alpha,
    computed via `dE_dalpha`. It applies a small step in the negative gradient direction
    to iteratively reach the optimal alpha value that minimizes the energy.

    Parameters
    ----------
    position_vec : numpy.ndarray
        Array of walker positions after equilibration.
    alpha : float
        Current value of the variational parameter alpha.
    learning_rate : float
        Step size for updating alpha.

    Returns
    -------
    tuple
        Updated alpha value (float) and the computed derivative dE/dα (float).
    
    Raises
    ------
    ValueError
        If `learning_rate` is negative.

    Warnings
    --------
    UserWarning
        If `learning_rate` is zero, meaning alpha remains unchanged.

    Notes
    -----
    - The function updates alpha using the formula:
      α_new = α - γ * dE/dα
    - The learning rate γ controls the step size in optimization:
      - If γ is too small, convergence will be too slow.
      - If γ is too large, the optimization may fail to converge properly.
    - This method ensures that alpha dynamically adjusts during the Metropolis simulation,
      improving efficiency in finding the optimal wavefunction parameters.
    """

    if learning_rate < 0:
        raise ValueError(
            "Learning rate must be a non-negative number, otherwise it would cause divergence."
        )

    if learning_rate == 0:
        warnings.warn(
            "The learning rate is set to 0. No optimization will be performed for alpha.",
            UserWarning,
        )

    dE_da = dE_dalpha(position_vec, alpha)
    alpha = alpha - learning_rate * dE_da
    return alpha, dE_da


def metropolis(equilibration_steps, numsteps, numwalkers, alpha, learning_rate):
    """
    Perform Metropolis-Hastings sampling to optimize walker positions and the variational parameter alpha.

    This function implements the Metropolis-Hastings algorithm to generate walker positions that
    follow the probability distribution defined by the trial wavefunction. Additionally, it
    optimizes alpha on-the-fly using a gradient descent approach.

    Parameters
    ----------
    equilibration_steps : int
        Number of thermalization steps performed before measuring properties. Must be non-negative.
    numsteps : int
        Number of Metropolis sampling steps after equilibration. Must be a positive integer.
    numwalkers : int
        Number of walkers used to explore phase space. Must be a positive integer.
    alpha : float
        Initial value of the variational parameter alpha. It is updated iteratively to minimize energy.
    learning_rate : float
        Step size for updating alpha.

    Returns
    -------
    tuple
        position_vec : numpy.ndarray
            Final walker positions after the Metropolis simulation.
        alpha : float
            Final optimized value of alpha.
        alpha_buffer : numpy.ndarray
            Array storing the evolution of alpha over the simulation steps.
        E_buffer : numpy.ndarray
            Array storing the evolution of the mean local energy over time.
        dE_da_buffer : numpy.ndarray
            Array storing the evolution of the derivative of energy with respect to alpha.
        initial_pos : numpy.ndarray
            Initial walker positions before the Metropolis updates.

    Raises
    ------
    TypeError
        If any of `equilibration_steps`, `numsteps`, or `numwalkers` is not an integer.
    ValueError
        If `numsteps` or `numwalkers` is less than or equal to zero.
        If `equilibration_steps` is negative.
        If division by zero is encountered in the probability calculation.

    Notes
    -----
    - **Initialization**: Walkers are initialized with uniformly distributed positions between 2 and 3.
    - **Thermalization**: `equilibration_steps` define the number of moves performed before measurements start.
    - **Metropolis Algorithm**:
        - Each walker is displaced by adding a small Gaussian-distributed random shift.
        - The move is accepted with probability p = Ψ(new_position) / Ψ(old_position), ensuring efficient sampling.
    - **Alpha Optimization**: After every sampling step, alpha is updated using `alpha_opt_on_fly`.
    - **Typical Parameters**:
        - `numwalkers = 5000`: Large number of walkers ensures statistical accuracy.
        - `numsteps = 120`: Steps performed after equilibration.
        - `equilibration_steps = 3000`: Ensures the system stabilizes before collecting data.
        - `alpha ≈ 0.8`: Initial guess, typically converging toward the optimal value (≈0.5 for hydrogen atom).
        - `learning_rate = 0.01`: Controls the step size in alpha optimization.
    """

    if (
        not isinstance(equilibration_steps, int)
        or not isinstance(numsteps, int)
        or not isinstance(numwalkers, int)
    ):
        raise TypeError(
            "equilibration_steps, numsteps, and numwalkers must be integers."
        )

    if numsteps <= 0 or numwalkers <= 0:
        raise ValueError(
            "numsteps and numwalkers must be positive integers greater than 0."
        )

    if equilibration_steps < 0:
        raise ValueError(
            "equilibration_steps must be a non-negative integer."
        )

    if equilibration_steps == 0:
        warnings.warn(
            "equilibration_steps is set to 0. The system will not equilibrate before optimization.",
            UserWarning,
        )

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
                raise ValueError("Division by zero detected in p calculation.")
            p = numerator / denominator
            rand_unif_array = np.random.uniform(size=len(p))
            position_vec = np.where(p > rand_unif_array, new_position_vec, position_vec)

        alpha, dE_da = alpha_opt_on_fly(position_vec, alpha, learning_rate)
        alpha_buffer[j] = alpha
        dE_da_buffer[j] = dE_da
        E_buffer[j] = np.mean(local_energy_func(position_vec, alpha))

    return position_vec, alpha, alpha_buffer, E_buffer, dE_da_buffer, initial_pos
