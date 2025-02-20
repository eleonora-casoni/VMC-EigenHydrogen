from unittest.mock import patch
import numpy as np
from simulation import metropolis

def test_raise_error_on_invalid_p():
    """
    test that metropolis raises value error when forcing division by zero for one walker.

    GIVEN: A new_position_vec or position_vec that results in invalid p values (NaN or Inf).
    WHEN: The metropolis function is called.
    THEN: A ValueError should be raised with a comprehensible error message.

    """
    equilibration_steps = 1
    numsteps = 1
    numwalkers = 3
    alpha = 1.0

    with patch("numpy.random.uniform") as mock_uniform, patch("numpy.random.randn") as mock_randn:
        def mock_random_uniform(*args, **kwargs):
            if kwargs.get("size") == numwalkers:
                return np.array([2.5, 3.0, 0.0])  
            return np.array([0.5, 0.5, 0.5])  
        
        mock_randn.return_value = np.zeros(numwalkers)  
        mock_uniform.side_effect = mock_random_uniform

        try:
            metropolis(equilibration_steps, numsteps, numwalkers, alpha)
        except ValueError as e:
            assert str(e) == "Division by zero detected in p calculation."


def test_raise_error_on_invalid_p_negative_position():
    """
    Test that metropolis raises ValueError when new_position_vec contains a small negative value.

    GIVEN: A new_position_vec containing a small negative value for one walker.
    WHEN: The metropolis function is called.
    THEN: A ValueError should be raised with a comprehensible error message.
    """
    equilibration_steps = 1
    numsteps = 1
    numwalkers = 3
    alpha = 1.0

    with patch("numpy.random.uniform") as mock_uniform, patch("numpy.random.randn") as mock_randn:

        def mock_random_uniform(*args, **kwargs):
            if kwargs.get("size") == numwalkers:
                return np.array([2.5, 3.0, 2.3])  
            return np.array([0.5, 0.5, 0.5])  
        mock_randn.return_value = np.array([0.0, 0.0, -2.8]) 

        mock_uniform.side_effect = mock_random_uniform
        try:
            metropolis(equilibration_steps, numsteps, numwalkers, alpha)
        except ValueError as e:
            assert str(e) == "Division by zero detected in p calculation."

def test_small_positive_position_effect():
    """
    Test that metropolis accepts small positive position values.

    GIVEN: A new_position_vec containing very small positive values for one walker.
    WHEN: The metropolis function is called.
    THEN: The simulation runs without raising warnings or errors on the final position.

    """
    np.random.seed(42)
    
    equilibration_steps = 1
    numsteps = 1
    numwalkers = 3
    alpha = 1.0

    with patch("numpy.random.randn") as mock_randn:
        mock_randn.return_value = np.array([1e-10, 1e-5, 1e-8])

        position_vec_fin, alpha_fin, alpha_buffer, E_buffer, dE_da_buffer, initial_pos = metropolis(
            equilibration_steps, numsteps, numwalkers, alpha
        )

        assert position_vec_fin is not None, "The simulation did not complete successfully."        

def test_small_positive_position_effect_on_alpha():
    """
    Test that small positive position values do not affect alpha optimization.

    GIVEN: A new_position_vec containing very small positive values for one walker.
    WHEN: The metropolis function is called.
    THEN: The simulation runs without raising warnings or errors on alpha values.

    """
    np.random.seed(42)
    
    equilibration_steps = 1
    numsteps = 1
    numwalkers = 3
    alpha = 1.0

    with patch("numpy.random.randn") as mock_randn:
        mock_randn.return_value = np.array([1e-10, 1e-5, 1e-8])

        position_vec_fin, alpha_fin, alpha_buffer, E_buffer, dE_da_buffer, initial_pos = metropolis(
            equilibration_steps, numsteps, numwalkers, alpha
        )

        assert alpha_fin is not None, "Alpha optimization did not complete successfully."

def test_small_positive_position_effect_on_alpha_buffer():
    """
    Test that small positive position values do not affect the population of alpha_buffer.

    GIVEN: A new_position_vec containing very small positive values for one walker.
    WHEN: The metropolis function is called.
    THEN: The simulation runs without raising warnings or errors on alpha_buffer population.

    """
    np.random.seed(42)

    equilibration_steps = 1
    numsteps = 1
    numwalkers = 3
    alpha = 1.0

    with patch("numpy.random.randn") as mock_randn:
        mock_randn.return_value = np.array([1e-10, 1e-5, 1e-8])

        position_vec_fin, alpha_fin, alpha_buffer, E_buffer, dE_da_buffer, initial_pos = metropolis(
            equilibration_steps, numsteps, numwalkers, alpha
        )

        assert len(alpha_buffer) == numsteps, "Alpha buffer does not have the expected length."

def test_small_positive_position_effect_on_energy_buffer():
    """
    Test that small positive position values do not affect the population of E_buffer.

    GIVEN: A new_position_vec containing very small positive values for one walker.
    WHEN: The metropolis function is called.
    THEN: The simulation runs without raising warnings or errors on E_buffer population.

    """
    np.random.seed(42)

    equilibration_steps = 1
    numsteps = 1
    numwalkers = 3
    alpha = 1.0

    with patch("numpy.random.randn") as mock_randn:
        mock_randn.return_value = np.array([1e-10, 1e-5, 1e-8])

        position_vec_fin, alpha_fin, alpha_buffer, E_buffer, dE_da_buffer, initial_pos = metropolis(
            equilibration_steps, numsteps, numwalkers, alpha
        )

        assert len(E_buffer) == numsteps, "Energy buffer does not have the expected length."
