import numpy as np
import pytest
from vmc_simulation.simulation import trial_wavefunction

def test_negative_positions():
    """
    Test that trial_wavefunction returns 0 for negative positions.

    GIVEN: An array of negative position values.
    WHEN: The trial_wavefunction function is called with these positions and any alpha value.
    THEN: The function returns an array of zeros with the same shape as the input.

    """
    x_negative = np.array([-1.0, -5.0, -0.001, -100.0])
    alpha = 1.0  
    expected_output = np.zeros_like(x_negative)
    result = trial_wavefunction(x_negative, alpha)

    assert np.array_equal(result, expected_output), "The function did not return 0 for negative positions."

def test_zero_position():
    """
    Test that trial_wavefunction returns 0 for position = 0.

    GIVEN: A position array containing zero.
    WHEN: The trial_wavefunction function is called with this position and any alpha value.
    THEN: The function returns 0 for the position = 0.
    """
    x_zero = np.array([0])  
    alpha = 1.0  
    result = trial_wavefunction(x_zero, alpha)
    expected_output = np.array([0])  

    assert np.array_equal(result, expected_output), "The function did not return 0 for position = 0."

def test_large_positive_x():
    """
    Test that the trial_wavefunction returns a value close to 0 for large positive x.

    GIVEN: A large positive value for x.
    WHEN: The trial_wavefunction function is called with this position and any alpha value.
    THEN: The function should return a value close to 0.
    """
    x_large = 1e10  
    alpha = 1.0  
    result = trial_wavefunction(x_large, alpha)

    assert np.isclose(result, 0, atol=1e-10), "The function did not return a value close to 0 for large positive x."

def test_small_positive_x():
    """
    Test that the trial_wavefunction returns a value close to 1 for small positive x.

    GIVEN: A small positive value for x.
    WHEN: The trial_wavefunction function with this position and any alpha value.
    THEN: The function should return a value close to 1.
    """
    x_small = 1e-10  
    alpha = 1.0  
    result = trial_wavefunction(x_small, alpha)

    assert np.isclose(result, 1, atol=1e-10), "The function did not return a value close to 1 for small positive x."

def test_uniform_x_between_2_and_3():
    """
    Test that trial_wavefunction returns correct values for x uniformly distributed between 2 and 3 (typical value reange).

    GIVEN: An array of x values uniformly distributed between 2 and 3.
    WHEN: The trial_wavefunction function is called with these inputs and any alpha value.
    THEN: The function should return exp(-alpha * x) for all values of x in this range.
    """
    x_uniform = np.random.uniform(low=2.0, high=3.0, size=1000)  
    alpha = 1.0  
    result = trial_wavefunction(x_uniform, alpha)
    expected_output = np.exp(-alpha * x_uniform)

    assert np.allclose(result, expected_output, atol=1e-10), "The function did not return correct exponential values for x uniformly distributed between 2 and 3."

def test_output_array_length():
    """
    Test that trial_wavefunction returns an output array with the same length as the input array.

    GIVEN: An array of x values uniformly distributed between 2 and 3.
    WHEN: The trial_wavefunction function is called with these inputs and any alpha value.
    THEN: The function should return an output array with the same length as the input array.
    """
    x_uniform = np.random.uniform(low=2.0, high=3.0, size=1000)  
    alpha = 1.0  
    result = trial_wavefunction(x_uniform, alpha)

    assert len(result) == len(x_uniform), "The output array does not have the same length as the input array."

def test_zero_alpha():
    """
    Test that trial_wavefunction returns 1 for when alpha = 0.

    GIVEN: alpha = 0.
    WHEN: The trial_wavefunction function is called with any positive x.
    THEN: The function should return 1 for all positive x.
    """
    x_values = np.array([0.1, 1.0, 10.0, 100.0])  
    alpha = 0.0  
    result = trial_wavefunction(x_values, alpha)
    expected_output = np.ones_like(x_values)  

    assert np.array_equal(result, expected_output), "The function did not return 1 when alpha = 0."

def test_small_negative_alpha():
    """
    Test that trial_wavefunction returns values greater than 1 for small negative alpha.

    GIVEN: A small negative value for alpha.
    WHEN: The trial_wavefunction function is called with any positive x.
    THEN: The function should return values greater than 1.
    """
    x_values = np.array([0.1, 1.0, 10.0, 100.0])  
    alpha = -0.01  
    result = trial_wavefunction(x_values, alpha)
    expected_output = np.exp(-alpha * x_values)  

    assert np.allclose(result, expected_output, atol=1e-10), "The function did not return the expected exponential values."

def test_large_negative_alpha():
    """
    Test that trial_wavefunction raises a ValueError for extremely large negative alpha.

    GIVEN: A very large negative alpha value.
    WHEN: The trial_wavefunction function is called with any positive x.
    THEN: The function should raise a ValueError to prevent numerical instability.
    """
    x_values = np.array([1.0, 2.0, 3.0])  
    alpha = -10000.0  

    with pytest.raises(ValueError, match="Too large negative alpha causes numerical instability."):
        trial_wavefunction(x_values, alpha)

def test_large_positive_alpha():
    """
    Test that trial_wavefunction raises a ValueError for excessively large alpha.

    GIVEN: A very large positive alpha.
    WHEN: The trial_wavefunction function is called with any reasonable positive x.
    THEN: The function should raise a ValueError to prevent numerical instability.
    """
    x_values = np.array([1.0, 2.0, 3.0])  
    alpha = 300  

    with pytest.raises(ValueError, match="Too large alpha causes numerical instability."):
        trial_wavefunction(x_values, alpha)

def test_small_positive_alpha():
    """
    Test that trial_wavefunction returns values close to 1 for very small positive alpha.

    GIVEN: A very small positive alpha value.
    WHEN: The trial_wavefunction function is called with any positive x.
    THEN: The function should return values close to 1.
    """
    x_values = np.array([0.1, 1.0, 5.0, 10.0])  
    alpha = 1e-6  

    result = trial_wavefunction(x_values, alpha)
    
    assert np.allclose(result, 1, atol=1e-6), "The function did not return values close to 1 for small positive alpha."

@pytest.mark.parametrize("invalid_alpha", [ [1.0, 2.0], "invalid_value", 2 + 3j])
def test_invalid_alpha_types(invalid_alpha):
    """
    Test that trial_wavefunction raises TypeError for invalid alpha types.

    GIVEN: Invalid type inputs for alpha.
    WHEN: The trial_wavefunction function is called with reasonable positions.
    THEN: A TypeError should be raised with the correct message.
    """
    x_values = np.array([1.0, 2.0, 3.0])

    with pytest.raises(TypeError, match="Alpha must be a real number."):
        trial_wavefunction(x_values, invalid_alpha)
