import numpy as np
import pytest
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

def test_learning_rate_typical():
    """
    Test that alpha_opt_on_fly updates correctly with a typical learning rate.

    GIVEN: A standard position vector, alpha, and a reasonable learning rate.
    WHEN: The function is called.
    THEN: Alpha should update by the expected amount.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1.0
    learning_rate = 0.01

    dE_da = dE_dalpha(position_vec, alpha)
    expected_alpha = alpha - learning_rate * dE_da
    new_alpha, computed_dE_da = alpha_opt_on_fly(position_vec, alpha, learning_rate)

    assert np.isclose(new_alpha, expected_alpha, atol=1e-10), f"Expected {expected_alpha}, got {new_alpha}"

def test_small_learning_rate():
    """
    Test that alpha_opt_on_fly updates slightly when learning rate is very small.

    GIVEN: A standard position vector, alpha, and a very small learning rate.
    WHEN: The function is called.
    THEN: Alpha should update, but by a very small amount.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1.0
    learning_rate = 1e-6

    dE_da = dE_dalpha(position_vec, alpha)
    expected_alpha = alpha - learning_rate * dE_da
    new_alpha, computed_dE_da = alpha_opt_on_fly(position_vec, alpha, learning_rate)

    assert np.isclose(new_alpha, expected_alpha, atol=1e-10), f"Expected {expected_alpha}, got {new_alpha}"

def test_large_learning_rate():
    """
    Test that alpha_opt_on_fly updates significantly when learning rate is large.

    GIVEN: A standard position vector, alpha, and a large learning rate.
    WHEN: The function is called.
    THEN: Alpha should update by a large amount.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1.0
    learning_rate = 5.0 

    dE_da = dE_dalpha(position_vec, alpha)
    expected_alpha = alpha - learning_rate * dE_da
    new_alpha, computed_dE_da = alpha_opt_on_fly(position_vec, alpha, learning_rate)

    assert np.isclose(new_alpha, expected_alpha, atol=1e-10), f"Expected {expected_alpha}, got {new_alpha}"

def test_alpha_remains_constant_with_zero_learning_rate():
    """
    Test that when learning_rate is set to 0, alpha does not change.

    GIVEN: A standard position vector and alpha with learning_rate = 0.
    WHEN: The function is called.
    THEN: Alpha should remain unchanged, and a warning should be raised.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1.0
    learning_rate = 0.0

    with pytest.warns(UserWarning, match="The learning rate is set to 0. No optimization will be performed for alpha."):
        new_alpha, dE_da = alpha_opt_on_fly(position_vec, alpha, learning_rate)

    assert new_alpha == alpha, f"Expected alpha to remain {alpha}, but got {new_alpha}"

def test_negative_learning_rate():
    """
    Test that a negative learning rate raises an error.

    GIVEN: A position vector, alpha, and a negative learning rate.
    WHEN: The function is called.
    THEN: It should raise a ValueError or warning.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1.0
    learning_rate = -0.01  

    with pytest.raises(ValueError, match="Learning rate must be a non-negative number, otherwise it would cause divergence."):
        alpha_opt_on_fly(position_vec, alpha, learning_rate)

@pytest.mark.parametrize("invalid_learning_rate", ["string", None, [], {}, 2 + 3j])
def test_invalid_learning_rate_type(invalid_learning_rate):
    """
    Test that alpha_opt_on_fly raises TypeError for invalid learning_rate types.

    GIVEN: A valid position vector and alpha.
    WHEN: An invalid learning_rate type is passed (string, None, list, dict, bool).
    THEN: A TypeError should be raised.
    """
    position_vec = np.array([1.0, 2.0, 3.0])
    alpha = 1.0  

    with pytest.raises(TypeError, match="learning_rate must be a float or an integer."):
        alpha_opt_on_fly(position_vec, alpha, invalid_learning_rate)

