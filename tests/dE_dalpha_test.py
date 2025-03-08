import numpy as np
from vmc_simulation.simulation import dE_dalpha
from vmc_simulation.simulation import local_energy_func

def test_output_is_scalar():
    """
    Test that dE_dalpha returns a scalar value.

    GIVEN: An array of valid positions x and a valid alpha.
    WHEN: The function is called.
    THEN: The output should be a single scalar value.
    """
    x_values = np.linspace(0.5, 3, 100)  
    alpha = 1.0  
    result = dE_dalpha(x_values, alpha)
    
    assert np.isscalar(result), "dE_dalpha should return a scalar value."

def test_typical_values():
    """
    Test dE_dalpha with typical input values.

    GIVEN: A standard array of x positions and a reasonable alpha.
    WHEN: The function is called.
    THEN: The returned value should be a finite, reasonable value.
    """
    x_values = np.array([1.0, 2.0, 3.0])
    alpha = 1.0

    El = local_energy_func(x_values, alpha)
    ln_wf = -x_values
    expected_value = 2 * (np.mean(El * ln_wf) - np.mean(El) * np.mean(ln_wf))
    result = dE_dalpha(x_values, alpha)

    assert np.isfinite(result), "dE_dalpha should return a finite number for typical values."
    assert np.isclose(result, expected_value, atol=1e-10), f"Expected {expected_value}, got {result}"

def test_small_positive_alpha():
    """
    Test dE_dalpha behavior with a very small positive alpha value.

    GIVEN: An array of valid positions x and a small positive alpha.
    WHEN: The function is called.
    THEN: The function should return a finite number.
    """
    x_values = np.linspace(0.5, 3, 100)
    alpha = 1e-10

    El = local_energy_func(x_values, alpha)
    ln_wf = -x_values
    expected_value = 2 * (np.mean(El * ln_wf) - np.mean(El) * np.mean(ln_wf))
    result = dE_dalpha(x_values, alpha)

    assert np.isfinite(result), "dE_dalpha should return a finite number for small positive alpha."
    assert np.isclose(result, expected_value, atol=1e-10), f"Expected {expected_value}, got {result}"

def test_large_positive_alpha():
    """
    Test dE_dalpha does not overflow with a very large alpha value.

    GIVEN:  An array of valid positions x and a large alpha.
    WHEN: The function is called.
    THEN: The function should return a finite number without overflow.
    """
    x_values = np.linspace(0.5, 3, 100)
    alpha = 1000
    result = dE_dalpha(x_values, alpha)

    El = local_energy_func(x_values, alpha)
    ln_wf = -x_values
    expected_value = 2 * (np.mean(El * ln_wf) - np.mean(El) * np.mean(ln_wf))

    assert np.isfinite(result), "dE_dalpha should not overflow for large alpha."
    assert np.isclose(result, expected_value, atol=1e-10), f"Expected {expected_value}, got {result}"

def test_small_negative_alpha():
    """
    Test dE_dalpha behavior with a small negative alpha.

    GIVEN: An array of valid positions x and a small negative alpha.
    WHEN: The function is called.
    THEN: The function should return a finite number.
    """
    x_values = np.linspace(2, 3, 100)
    alpha = -1e-10

    El = local_energy_func(x_values, alpha)
    ln_wf = -x_values
    expected_value = 2 * (np.mean(El * ln_wf) - np.mean(El) * np.mean(ln_wf))
    result = dE_dalpha(x_values, alpha)

    assert np.isfinite(result), "dE_dalpha should return a finite number for small negative alpha."
    assert np.isclose(result, expected_value, atol=1e-10), f"Expected {expected_value}, got {result}"

def test_small_positive_x():
    """
    Test dE_dalpha behaviour with small positive x values.

    GIVEN: Very small positive x values approaching zero and any alpha value.
    WHEN: The function is called.
    THEN: The function should return a finite number.
    """
    x_values = np.array([1e-10, 1e-8, 1e-5])  
    alpha = 1.0

    El = local_energy_func(x_values, alpha)
    ln_wf = -x_values
    expected_value = 2 * (np.mean(El * ln_wf) - np.mean(El) * np.mean(ln_wf))
    result = dE_dalpha(x_values, alpha)

    assert np.isfinite(result), "dE_dalpha should not result in NaN or Inf for small x values."
    assert np.isclose(result, expected_value, atol=1e-10), f"Expected {expected_value}, got {result}"

def test_large_x_values():
    """
    Test dE_dalpha with very large positive x values.

    GIVEN: An array of very large positive x values and any alpha value.
    WHEN: The function is called.
    THEN: The function should return a finite number without overflow.
    """
    x_values = np.array([1e6, 1e8, 1e10])
    alpha = 1.0

    El = local_energy_func(x_values, alpha)
    ln_wf = -x_values
    expected_value = 2 * (np.mean(El * ln_wf) - np.mean(El) * np.mean(ln_wf))
    result = dE_dalpha(x_values, alpha)

    assert np.isfinite(result), "dE_dalpha should not overflow for large x values."
    assert np.isclose(result, expected_value, atol=1e-10), f"Expected {expected_value}, got {result}"





