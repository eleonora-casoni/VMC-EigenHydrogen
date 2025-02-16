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
