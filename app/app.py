# pylint: disable=no-member
import argparse
from flask import Flask, request, jsonify
import sys

# Local imports
from lambdaml.core.deploy import ModelWrapper
from lambdaml.util import load_yaml
from lambdaml.log import set_global_log_level

argparser = argparse.ArgumentParser(description="Deploy model from registry with flask")

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
    default=os.environ["LAMBDAML_REGISTRY"] if "LAMBDAML_REGISTRY" in os.environ else "app/registry",
    help="Flask app directory where model state dictionaries and parameters will be copied",
)

argparser.add_argument(
    "--flask_host",
    type=str,
    default="0.0.0.0",
    help="Flask app host name",
)

argparser.add_argument(
    "--flask_port",
    type=int,
    default=5000,
    help="Flask app port",
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
trials_to_codenames = load_json(metadata_path, trial_to_codenames)
codenames_to_trials = {trials_to_codenames[key]:key for key in trials_to_codenames}

# Load the appropriate trial id
if not args["trial_id"] in trials_to_codenames:

    # Check if you were passed the literal trial id and reset to the codename
    if args["trial_id"] in codenames_to_trials:
        args["trial_id"] = codenames_to_trials[args["trial_id"]]
    
    # Otherwise default to just printing out the available codenames
    else:
        print("Requested trial id not available from registry in:\n\t", metadata_dir)
        print("trial uuid =>\t codename")
        print("--------------------------------------------------")
        for trial in trials_to_codenames:
            print("\t",trial,"=>\t",trials_to_codenames[trial])
        sys.exit(0)

# Initialialize flask app and model
app = Flask(__name__)
model = ModelWrapper(
    registry=args["registry"],
    study_name=args["optuna_study_name"],
    trial_id=args["trial_id"],
)

# Define the app
@app.route("/predict", methods=["POST"])
def predict():
    bank_tables = request.get_json()
    try:
        prob = model.predict(bank_tables)
        return jsonify({"probability": prob})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the flaskapp with the specified
app.run(host=args["flask_host"], port=args["flask_port"])
