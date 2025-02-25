import numpy as np
from unittest.mock import patch
from simulation import alpha_opt_on_fly

def test_large_positions():
    """
    Test that alpha_opt_on_fly behaves correctly for large position values.

    GIVEN: A position vector with large values and any alpha value.
    WHEN: The function is called.
    THEN: The function should return an updated alpha without overflow.
    """
    position_vec = np.array([1e6, 2e6, 3e6])  
    alpha = 1.0  
    new_alpha, dE_da = alpha_opt_on_fly(position_vec, alpha)

    assert np.isfinite(new_alpha), "Updated alpha should be a finite number."
    assert np.isfinite(dE_da), "dE/da should be a finite number."

def test_small_alpha():
    """
    Test that alpha_opt_on_fly behaves well for small positive alpha.

    GIVEN: A reasonable position vector and a very small alpha.
    WHEN: The function is called.
    THEN: The function should return an updated alpha without underflow.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1e-10  
    new_alpha, dE_da = alpha_opt_on_fly(position_vec, alpha)

    assert np.isfinite(new_alpha), "Updated alpha should be a finite number."
    assert np.isfinite(dE_da), "dE/da should be a finite number."

def test_large_alpha():
    """
    Test that alpha_opt_on_fly behaves correctly for large alpha.

    GIVEN: A reasonable position vector and a very large alpha.
    WHEN: The function is called.
    THEN: The function should return an updated alpha without overflow.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1e6  
    new_alpha, dE_da = alpha_opt_on_fly(position_vec, alpha)

    assert np.isfinite(new_alpha), "Updated alpha should be a finite number."
    assert np.isfinite(dE_da), "dE/da should be a finite number."

def test_alpha_update_behavior():
    """
    Test that alpha updates when dE_da is nonzero and remains unchanged when dE_da is zero.

    GIVEN: A position vector and alpha.
    WHEN: The function is called.
    THEN: If dE_da is zero, alpha should not change. Otherwise, it should update.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1.0

    new_alpha, dE_da = alpha_opt_on_fly(position_vec, alpha)
    if dE_da == 0:
        assert new_alpha == alpha, "Alpha should remain unchanged if dE_da is zero."
    else:
        assert np.isfinite(new_alpha), "Updated alpha should be a finite number."
        assert np.isfinite(dE_da), "dE/da should be a finite number."
        assert new_alpha != alpha, "Alpha should update when dE_da is nonzero."

