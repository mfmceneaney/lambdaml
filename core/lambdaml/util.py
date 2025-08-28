# UTIL
import yaml
import json
import argparse


def save_yaml(path, data):
    """
    Parameters
    ----------
    path : str, required
        Path to yaml file
    data : dict, required
        Dictionary to save

    Description
    -----------
    Save a dictionary to a yaml file.
    """

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(path):
    """
    Parameters
    ----------
    path : str, required
        Path to yaml file

    Returns
    -------
    dict
        dictionary of yaml contents

    Description
    -----------
    Load a yaml from a file.
    """

    yaml_args = {}
    with open(path, encoding="utf-8") as f:
        yaml_args = yaml.safe_load(f)
    return yaml_args


def save_json(path, data):
    """
    Parameters
    ----------
    path : str, required
        Path to json file
    data : dict, required
        Dictionary to save

    Description
    -----------
    Save a dictionary to a json file.
    """

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    """
    Parameters
    ----------
    path : str, required
        Path to json file

    Returns
    -------
    dict
        dictionary of json contents

    Description
    -----------
    Load a json from a file.
    """

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def float_or_choice(value, choices=("auto", "default", "none")):
    try:
        # Try converting to float
        return float(value)
    except ValueError as exc:
        # If not a float, check if it's an allowed string
        if value.lower() in choices:
            return value.lower()
        raise argparse.ArgumentTypeError(
            f"Value must be a float or one of: {', '.join(choices)}"
        ) from exc
