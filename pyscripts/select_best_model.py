# pylint: disable=no-member
import argparse
import os.path as osp

# Local imports
from lambdaml.deploy import select_best_model
from lambdaml.log import set_global_log_level
from lambdaml.util import load_yaml


argparser = argparse.ArgumentParser(description="Select best model from Optuna database")

argparser.add_argument(
    "--log_level",
    type=str,
    default="INFO",
    choices=["debug", "info", "warning", "error", "critical", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Log level",
)

argparser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to config yaml",
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
    "--gnn_server_dir",
    type=str,
    default="gnn_server/",
    help="GNN server directory",
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

# Select best model and copy model state 
# and parameter definitions to GNN server directory
select_best_model(**args)
