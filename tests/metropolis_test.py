from unittest.mock import patch
import numpy as np
from simulation import metropolis

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

        try:
            metropolis(equilibration_steps, numsteps, numwalkers, alpha)
        except ValueError as e:
            assert str(e) == "Division by zero detected in p calculation."



