import pytest
from vmc_simulation.config_handler import parse_config_file

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
