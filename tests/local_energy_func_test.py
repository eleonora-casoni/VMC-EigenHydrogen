import numpy as np
from vmc_simulation.simulation import local_energy_func

def test_standard_positive_x():
    """
    Test that local_energy_func returns finite values for standard positive x.

    GIVEN: A set of standard positive x values.
    WHEN: The function is called with any alpha value.
    THEN: It should return finite energy values.
    """
    x_values = np.array([1.0, 2.0, 3.0, 0.5])
    alpha = 1.0  
    result = local_energy_func(x_values, alpha)

    assert np.all(np.isfinite(result)), "The function returned non-finite values for standard positive x."

def test_small_positive_x():
    """
    Test that local_energy_func does not crash for very small positive x.

    GIVEN: Very small positive x values.
    WHEN: The function is called with any alpha value.
    THEN: It should return finite values and not  raise an error.
    """
    x_values = np.array([1e-10, 1e-8, 1e-6])  
    alpha = 1.0  
    result = local_energy_func(x_values, alpha)

    assert np.all(np.isfinite(result)), "The function returned non-finite values for small positive x."

def test_large_positive_x():
    """
    Test that local_energy_func returns expected values for large x.

    GIVEN: Large x values.
    WHEN: The function is called with any alpha value.
    THEN: It should approach -alpha^2 / 2.
    """
    x_values = np.array([1e5, 1e10])  
    alpha = 2.0  
    expected_values = - (alpha**2) / 2  

    result = local_energy_func(x_values, alpha)
    
    assert np.allclose(result, expected_values, atol=1e-5), "The function does not behave as expected for large x."

def test_output_array_length():
    """
    Test that local_energy_func returns an output array of the same length as the input.

    GIVEN: An array of standard positive x values.
    WHEN: The local_energy_func function is called with a valid alpha.
    THEN: The output array should have the same length as the input array.
    """
    x_values = np.linspace(0.5, 5.0, num=100)  
    alpha = 1.0  
    result = local_energy_func(x_values, alpha)

    assert len(result) == len(x_values), "The output array length does not match the input array length."

def test_large_positive_alpha():
    """
    Test that local_energy_func returns correct values for large alpha.

    GIVEN: A large alpha value.
    WHEN: The function is called with any position values.
    THEN: The energy values should scale correctly.
    """
    x_values = np.array([1.0, 5.0])
    alpha = 100.0  
    result = local_energy_func(x_values, alpha)

    assert np.all(np.isfinite(result)), "The function returned non-finite values for large alpha."

def test_local_energy_negative_alpha():
    """
    Test that local_energy_func returns correct values for a reasonable negative alpha.

    GIVEN: A negative but reasonable alpha value.
    WHEN: The local_energy_func function is called with valid positive x values.
    THEN: The function should return finite, real energy values (no NaN or Inf).
    """
    x_values = np.array([1.0, 5.0])
    alpha = -1.0 

    result = local_energy_func(x_values, alpha)

    assert np.all(np.isfinite(result)), "Local energy function returned NaN or Inf values for negative alpha."

