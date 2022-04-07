"""
Provides all the misc helper functions
"""
import yaml


def load_yaml(filepath: str) -> dict:
    """This function loads in a yaml configuration file
    Args:
        filepath (str): configuration filepath

    Returns:
        dict: configuration dictionary
    """
    with open(filepath, "r", encoding="utf-8") as yaml_file:
        configuration = yaml.safe_load(yaml_file)
    return configuration
