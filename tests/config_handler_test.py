import pytest
import tempfile
import os
from configparser import MissingSectionHeaderError
from vmc_simulation.config_handler import parse_config_file


def create_temp_ini(content):
    """
    Create a temporary .ini configuration file for testing purposes.

    This helper function generates a temporary .ini file with the given content, 
    which can be used for unit tests that require reading from a configuration file.
    The file persists until manually deleted to allow proper testing.

    Parameters
    ----------
    content : str
        The content to write into the temporary .ini file.

    Returns
    -------
    str
        The file path of the created temporary .ini file.

    Notes
    -----
    - The file is **not automatically deleted** after use. It should be manually deleted 
      using `os.remove(filepath)` after testing.

    """
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".ini")
    temp.write(content.encode())
    temp.close()
    return temp.name

def test_parse_valid_config():
    """
    his test dynamically creates a temporary `.ini` file, reads it using 
    `parse_config_file`, and verifies that the extracted values match expectations.

    GIVEN: A valid `.ini` configuration file with properly formatted parameters.
    WHEN: The `parse_config_file` function is called with this file's path.
    THEN: It should return a dictionary with correctly parsed parameters, 
    converting numerical values to `int` or `float` as needed.
    """
    config_content = """ [Simulation]
     numwalkers = 5000
     numsteps = 200
     equilibration_steps = 3000
     alpha = 0.9
     learning_rate = 0.02
     step_size = 0.15
     output_dir = results
    """
    config_path = create_temp_ini(config_content)
    
    expected_output = {
        "numwalkers": 5000,
        "numsteps": 200,
        "equilibration_steps": 3000,
        "alpha": 0.9,
        "learning_rate": 0.02,
        "step_size": 0.15,
        "output_dir": "results"
    }
    
    result = parse_config_file(config_path)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    os.remove(config_path)

def test_parse_nonexistent_file():
    """
    Test that parse_config_file raises FileNotFoundError for a nonexistent file.

    GIVEN: A path to a configuration file that does not exist.
    WHEN: The parse_config_file function is called with this nonexistent file path.
    THEN: A FileNotFoundError should be raised with the appropriate error message.
    """

    config_path = "nonexistent_config.ini"
    with pytest.raises(FileNotFoundError, match=f"Configuration file '{config_path}' not found."):
        parse_config_file(config_path)
