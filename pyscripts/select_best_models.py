# pylint: disable=no-member
import argparse
import os
import os.path as osp

# Local imports
from lambdaml.deploy import select_best_models
from lambdaml.log import set_global_log_level
from lambdaml.util import load_yaml


argparser = argparse.ArgumentParser(
    description="Select model from Optuna database with SQL query"
)

argparser.add_argument(
    "--log_level",
    type=str,
    default="INFO",
    choices=[
        "debug",
        "info",
        "warning",
        "error",
        "critical",
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ],
    help="Log level",
)

argparser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to config yaml",
)

argparser.add_argument(
    "--n_best_trials",
    type=int,
    default=1,
    help="Storage URL for Optuna",
)

argparser.add_argument(
    "--optuna_storage_url",
    type=str,
    default="sqlite:///optuna_study.db",
    help="Storage URL for Optuna",
)

argparser.add_argument(
    "--optuna_study_name",
    type=str,
    default="model_hpo",
    help="Optuna study name",
)

argparser.add_argument(
    "--optuna_study_direction",
    type=str,
    default="maximize",
    choices=["minimize", "maximize"],
    help="Optuna study direction",
)

argparser.add_argument(
    "--registry",
    type=str,
    default=(
        os.environ["LAMBDAML_REGISTRY"]
        if "LAMBDAML_REGISTRY" in os.environ
        else "app/registry"
    ),
    help="Flask app directory where model state dictionaries and parameters will be copied",
)

argparser.add_argument(
    "--codename_separator",
    type=str,
    default="-",
    help="Codename separator",
)

argparser.add_argument(
    "--encoder_fname",
    type=str,
    default="encoder.pt",
    help="Encoder model state dictionary file name",
)

argparser.add_argument(
    "--encoder_params_fname",
    type=str,
    default="encoder_params.json",
    help="Encoder parameters json file name",
)

argparser.add_argument(
    "--clf_fname",
    type=str,
    default="clf.pt",
    help="Classifier model state dictionary file name",
)

argparser.add_argument(
    "--clf_params_fname",
    type=str,
    default="clf_params.json",
    help="Classifier parameters json file name",
)

# Parse arguments and initialize argument dictionary
args_raw = argparser.parse_args()
args = {}

# Check if arguments have been supplied by yaml
if args_raw.config is not None and osp.exists(args_raw.config):
    args = load_yaml(args_raw.config)

# Otherwise take them from command line
else:
    args = vars(args_raw)

# Set log level
set_global_log_level(args["log_level"])
args.pop("log_level")

# Remove config argument
args.pop("config")

# Select top n models and copy model states
# and parameter definitions to flask app directory
select_best_models(**args)
