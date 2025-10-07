# pylint: disable=no-member
import argparse
from flask import Flask, request, jsonify
import sys
import os
import os.path as osp
import subprocess

# Local imports
from lambdaml.deploy import ModelWrapper
from lambdaml.util import load_yaml, load_json
from lambdaml.log import set_global_log_level, setup_logger


logger = setup_logger(__name__)

argparser = argparse.ArgumentParser(description="Deploy model from registry with flask")

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
    "--trial_id",
    type=str,
    default="",
    help="Trial id or codename",
)

argparser.add_argument(
    "--optuna_study_name",
    type=str,
    default="model_hpo",
    help="Optuna study name",
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
    "--host",
    type=str,
    default="0.0.0.0",
    help="App host name",
)

argparser.add_argument(
    "--port",
    type=int,
    default=5000,
    help="App port",
)

argparser.add_argument(
    "--mode",
    type=str,
    default="dev",
    choices=["dev","development","prod","production"],
    help="App run mode",
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

# Load app metadata
metadata_dir = osp.join(args["registry"], args["optuna_study_name"])
metadata_path = osp.join(metadata_dir, "metadata.json")
logger.debug("Loading app metadata from %s", metadata_path)
trials_to_codenames = load_json(metadata_path)
codenames_to_trials = {trials_to_codenames[key]: key for key in trials_to_codenames}

# Load the appropriate trial id
if not args["trial_id"] in codenames_to_trials:

    # Check if you were passed the literal trial id and reset to the codename
    if args["trial_id"] in trials_to_codenames:
        args["trial_id"] = trials_to_codenames[args["trial_id"]]

    # Otherwise default to just printing out the available codenames
    else:
        print("Requested trial id not available from registry in:\n\t", metadata_dir)
        print("trial uuid =>\t codename")
        print("--------------------------------------------------")
        for trial in trials_to_codenames:
            print("\t", trial, "=>\t", trials_to_codenames[trial])
        sys.exit(0)

def create_app():
    # Initialialize flask app and model
    app = Flask(__name__)
    trial_dir = osp.join(metadata_dir, args["trial_id"])
    model = ModelWrapper(
        trial_dir=trial_dir,
    )
    
    # Define the app
    @app.route("/predict", methods=["POST"])
    def predict():
        bank_tables = request.get_json()
        try:
            prob = model.predict(bank_tables)
            return jsonify({"probability": prob})
        except TypeError as e:
            return jsonify({"error": str(e)}), 500

    return app

# Run in production mode
if args["mode"].lower() in ("production", "prod"):

    # Serve the flask app from gunicorn
    subprocess.run([
        "gunicorn",
        "app:create_app()",  # Call factory function
        "--bind", f"{args["host"]}:{args["port"]}"
    ])

# Or in development mode
else:
    # Initialialize flask app and model
    app = Flask(__name__)
    trial_dir = osp.join(metadata_dir, args["trial_id"])
    model = ModelWrapper(
        trial_dir=trial_dir,
    )

    # Define the app
    @app.route("/predict", methods=["POST"])
    def predict():
        bank_tables = request.get_json()
        try:
            prob = model.predict(bank_tables)
            return jsonify({"probability": prob})
        except TypeError as e:
            return jsonify({"error": str(e)}), 500

    # Run the flaskapp with the specified
    app.run(host=args["host"], port=args["port"])
