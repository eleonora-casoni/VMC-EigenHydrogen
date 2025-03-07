from unittest.mock import patch
import numpy as np
import pytest
from vmc_simulation.simulation import metropolis

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

        with pytest.raises(ValueError, match="Division by zero detected in p calculation."):
            metropolis(equilibration_steps, numsteps, numwalkers, alpha)

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
        assert alpha_fin is not None, "Alpha optimization did not complete successfully."
        assert len(alpha_buffer) == numsteps, "Alpha buffer does not have the expected length."
        assert len(E_buffer) == numsteps, "Energy buffer does not have the expected length."     

def test_alpha_convergence():
    """
    Test that alpha converges towards an expected value when given reasonable parameters

    GIVEN: reasonable starting parameters.
    WHEN: The metropolis function is run for sufficient steps.
    THEN: Alpha should change and move toward a stable value.
    """
    np.random.seed(42)

    equilibration_steps = 100
    numsteps = 50
    numwalkers = 500
    alpha = 0.8  

    _, alpha_fin, alpha_buffer, _, _, _ = metropolis(equilibration_steps, numsteps, numwalkers, alpha)

    assert abs(alpha_buffer[-1] - alpha_buffer[0]) > 0.01, "Alpha did not change significantly, indicating no optimization."
    assert np.isfinite(alpha_fin), "Final alpha is not finite."

def test_final_positions_distribution():
    """
    Test that the final walker positions are within a reasonable range when given reasonable parameters.

    GIVEN: Standard parameters.
    WHEN: The metropolis function is run.
    THEN: The final walker positions should be within a physically reasonable range.
    """
    np.random.seed(42)

    equilibration_steps = 100
    numsteps = 50
    numwalkers = 500
    alpha = 0.8  

    position_vec_fin, _, _, _, _, _ = metropolis(equilibration_steps, numsteps, numwalkers, alpha)

    assert np.all(position_vec_fin > 0), "Some walker positions are non-physical (≤ 0)."
    assert np.max(position_vec_fin) < 15, "Position values are unreasonably large."

@pytest.mark.parametrize("invalid_value", ["string", 3.5, 2 + 3j, [], {}])
def test_invalid_types_input_parameters(invalid_value):
    """
    Test that metropolis raises a TypeError when given non-integer inputs.
    
    GIVEN: A non-integer value for numwalkers, numsteps, or equilibration_steps.
    WHEN: The metropolis function is called with any alpha value.
    THEN: The function should raise a TypeError.
    """
    np.random.seed(42)
    alpha = 1.0 

    with pytest.raises(TypeError, match="equilibration_steps, numsteps, and numwalkers must be integers."):
        metropolis(invalid_value, 100, 500, alpha)  

    with pytest.raises(TypeError, match="equilibration_steps, numsteps, and numwalkers must be integers."):
        metropolis(100, invalid_value, 500, alpha)  

    with pytest.raises(TypeError, match="equilibration_steps, numsteps, and numwalkers must be integers."):
        metropolis(100, 100, invalid_value, alpha) 

@pytest.mark.parametrize("invalid_value", [0, -1, -100])
def test_invalid_numsteps_and_numwalkers(invalid_value):
    """
    Test that metropolis raises a ValueError when numsteps or numwalkers is zero or negative.

    GIVEN: A zero or negative value for numsteps or numwalkers.
    WHEN: The metropolis function is called with any alpha value.
    THEN: A ValueError should be raised with the correct error message.
    """
    np.random.seed(42)
    alpha = 1.0  

    with pytest.raises(ValueError, match="numsteps and numwalkers must be positive integers greater than 0."):
        metropolis(100, invalid_value, 500, alpha)  
    with pytest.raises(ValueError, match="numsteps and numwalkers must be positive integers greater than 0."):
        metropolis(100, 100, invalid_value, alpha) 

@pytest.mark.parametrize("invalid_value", [-1, -100])
def test_invalid_equilibration_steps(invalid_value):
    """
    Test that metropolis raises a ValueError when equilibration_steps is negative.

    GIVEN: A negative value for equilibration_steps.
    WHEN: The metropolis function is called.
    THEN: A ValueError should be raised with the correct error message.
    """
    np.random.seed(42)
    alpha = 1.0  

    with pytest.raises(ValueError, match="equilibration_steps must be a non-negative integer."):
        metropolis(invalid_value, 100, 500, alpha)  

def test_equilibration_zero_warning():
    """
    Test that metropolis raises a warning when equilibration_steps is set to 0.
    
    GIVEN: equilibration_steps = 0.
    WHEN: metropolis is called with other reasonable parameters.
    THEN: A UserWarning should be raised.
    """
    np.random.seed(42)

    with pytest.warns(UserWarning, match="equilibration_steps is set to 0. The system will not equilibrate before optimization."):
        metropolis(0, 10, 10, 1.0) 