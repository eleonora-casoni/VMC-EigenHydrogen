import numpy as np
from vmc_simulation.simulation import alpha_opt_on_fly
from vmc_simulation.simulation import dE_dalpha

def test_alpha_update_typical():
    """
    Test that alpha_opt_on_fly updates alpha correctly with typical values.

    GIVEN: A standard position vector and alpha.
    WHEN: The function is called.
    THEN: Alpha should update correctly.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1.0
    learning_rate = 0.01
    
    dE_da = dE_dalpha(position_vec, alpha)  
    expected_alpha = alpha - learning_rate * dE_da  
    new_alpha, computed_dE_da = alpha_opt_on_fly(position_vec, alpha, learning_rate)

    assert np.isfinite(new_alpha), "Alpha should be a finite number."
    assert np.isfinite(computed_dE_da), "dE/da should be a finite number."
    assert np.isclose(new_alpha, expected_alpha, atol=1e-10), f"Expected {expected_alpha}, got {new_alpha}"

def test_large_positions():
    """
    Test that alpha_opt_on_fly behaves correctly for large position values.

    GIVEN: A position vector with large values and any alpha value.
    WHEN: The function is called.
    THEN: The function should return an updated alpha without overflow.
    """
    position_vec = np.array([1e6, 2e6, 3e6])  
    alpha = 1.0  
    learning_rate = 0.01

    dE_da = dE_dalpha(position_vec, alpha)
    expected_alpha = alpha - learning_rate * dE_da
    new_alpha, dE_da = alpha_opt_on_fly(position_vec, alpha, learning_rate)

    assert np.isfinite(new_alpha), "Updated alpha should be a finite number."
    assert np.isfinite(dE_da), "dE/da should be a finite number."
    assert np.isclose(new_alpha, expected_alpha, atol=1e-10), f"Expected {expected_alpha}, got {new_alpha}"

def test_small_alpha():
    """
    Test that alpha_opt_on_fly behaves well for small positive alpha.

    GIVEN: A reasonable position vector and a very small alpha.
    WHEN: The function is called.
    THEN: The function should return an updated alpha without underflow.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1e-10  
    learning_rate = 0.01

    dE_da = dE_dalpha(position_vec, alpha)
    expected_alpha = alpha - learning_rate * dE_da  
    new_alpha, dE_da = alpha_opt_on_fly(position_vec, alpha, learning_rate)

    assert np.isfinite(new_alpha), "Updated alpha should be a finite number."
    assert np.isfinite(dE_da), "dE/da should be a finite number."
    assert np.isclose(new_alpha, expected_alpha, atol=1e-10), f"Expected {expected_alpha}, got {new_alpha}"

def test_large_alpha():
    """
    Test that alpha_opt_on_fly behaves correctly for large alpha.

    GIVEN: A reasonable position vector and a very large alpha.
    WHEN: The function is called.
    THEN: The function should return an updated alpha without overflow.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1e6  
    learning_rate = 0.01

    dE_da = dE_dalpha(position_vec, alpha)
    expected_alpha = alpha - learning_rate * dE_da  
    new_alpha, dE_da = alpha_opt_on_fly(position_vec, alpha, learning_rate)

    assert np.isfinite(new_alpha), "Updated alpha should be a finite number."
    assert np.isfinite(dE_da), "dE/da should be a finite number."
    assert np.isclose(new_alpha, expected_alpha, atol=1e-10), f"Expected {expected_alpha}, got {new_alpha}"

def test_alpha_update_behavior():
    """
    Test that alpha updates when dE_da is nonzero and remains unchanged when dE_da is zero.

    GIVEN: A position vector and alpha.
    WHEN: The function is called.
    THEN: If dE_da is zero, alpha should not change. Otherwise, it should update.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1.0
    learning_rate = 0.01

    new_alpha, dE_da = alpha_opt_on_fly(position_vec, alpha, learning_rate)
    if dE_da == 0:
        assert new_alpha == alpha, "Alpha should remain unchanged if dE_da is zero."
    else:
        assert np.isfinite(new_alpha), "Updated alpha should be a finite number."
        assert np.isfinite(dE_da), "dE/da should be a finite number."
        assert new_alpha != alpha, "Alpha should update when dE_da is nonzero."
