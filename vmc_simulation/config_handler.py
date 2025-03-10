import configparser
import os

def parse_config_file(config_path):
    """
    Reads a configuration file and extracts parameters.

    Parameters
    ----------
    config_path : str
        Path to the .ini configuration file.

    Returns
    -------
    dict
        A dictionary containing simulation parameters from the config file.
    
    Raises
    ------
    FileNotFoundError
        If the specified config file does not exist.

    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found."
        )
    
    config = configparser.ConfigParser()
    config.read(config_path)

    config_params = {}

    if "Simulation" in config:
        section = config["Simulation"]
        if "numwalkers" in section:
            config_params["numwalkers"] = int(section["numwalkers"])
        if "numsteps" in section:
            config_params["numsteps"] = int(section["numsteps"])
        if "equilibration_steps" in section:
            config_params["equilibration_steps"] = int(section["equilibration_steps"])
        if "alpha" in section:
            config_params["alpha"] = float(section["alpha"])
        if "learning_rate" in section:
            config_params["learning_rate"] = float(section["learning_rate"])
        if "step_size" in section:
            config_params["step_size"] = float(section["step_size"])
        if "output_dir" in section:
            config_params["output_dir"] = section["output_dir"]

    return config_params
