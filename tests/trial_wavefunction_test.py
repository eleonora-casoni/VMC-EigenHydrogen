import numpy as np
import pytest
from simulation import trial_wavefunction


def test_negative_positions():
    """
    Test that trial_wavefunction() returns 0 for negative positions.

    GIVEN: An array of negative position values.
    WHEN: The trial_wavefunction function is called with these positions and any alpha value.
    THEN: The function returns an array of zeros with the same shape as the input.

    """
    x_negative = np.array([-1.0, -5.0, -0.001, -100.0])
    alpha = 1.0  # Arbitrary positive alpha value
    expected_output = np.zeros_like(x_negative)
    result = trial_wavefunction(x_negative, alpha)

    assert np.array_equal(result, expected_output), "The function did not return 0 for negative positions."

def test_zero_position():
    """
    Test that trial_wavefunction() returns 0 for position = 0.

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
    WHEN: The trial_wavefunction function is called with these inputs and a any alpha value.
    THEN: The function should return an output array with the same length as the input array.
    """
    x_uniform = np.random.uniform(low=2.0, high=3.0, size=1000)  
    alpha = 1.0  
    result = trial_wavefunction(x_uniform, alpha)

    assert len(result) == len(x_uniform), "The output array does not have the same length as the input array."