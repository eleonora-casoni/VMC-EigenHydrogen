import numpy as np
import pytest
from simulation import trial_wavefunction


def test_trial_wavefunction_negative_positions():
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

def test_trial_wavefunction_zero_position():
    """
    Test that trial_wavefunction() returns 0 for position = 0.

    GIVEN: A position array containing zero.
    WHEN: The trial_wavefunction function is called with this position and any value of alpha.
    THEN: The function returns 0 for the position = 0.
    """
    x_zero = np.array([0])  
    alpha = 1.0  
    result = trial_wavefunction(x_zero, alpha)
    expected_output = np.array([0])  
    assert np.array_equal(result, expected_output), "The function did not return 0 for position = 0."
