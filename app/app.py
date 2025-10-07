# pylint: disable=no-member
import argparse
import sys
import os
import os.path as osp
import subprocess

# Local imports
from lambdaml.deploy import create_app
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
    choices=["dev", "development", "prod", "production"],
    help="App run mode: flask for development and gunicorn for production",
)

argparser.add_argument(
    "--workers",
    type=int,
    default=(
        os.environ["SLURM_CPUS_ON_NODE"] if "SLURM_CPUS_ON_NODE" in os.environ else 4
    ),
    help="(gunicorn) Number of workers",
)

argparser.add_argument(
    "--threads", type=int, default=4, help="(gunicorn) Threads per worker"
)

argparser.add_argument(
    "--timeout", type=int, default=120, help="(gunicorn) Request timeout in seconds"
)

argparser.add_argument(
    "--backlog", type=int, default=2048, help="(gunicorn) Max pending connections"
)

argparser.add_argument(
    "--worker-class",
    type=str,
    default="gthread",
    help="(gunicorn) Worker class",
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

logger.debug("Original args = %s", args)

# Set log level
set_global_log_level(args["log_level"])
log_level = args.pop("log_level")

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

# Set config path for app called in create_app
os.environ["APP_CONFIG_PATH"] = osp.join(metadata_dir, args["trial_id"])

logger.debug("Updated args = %s", args)

# Run in production mode
if args["mode"].lower() in ("production", "prod"):

    # Serve the flask app from gunicorn
    subprocess.run(
        [
            "gunicorn",
            "lambdaml.wsgi:app",
            "--bind",
            f"{args["host"]}:{args["port"]}",
            "--workers",
            f"{args["workers"]}",  # Number of worker processes
            "--threads",
            f"{args["threads"]}",  # Threads per worker
            "--worker-class",
            f"{args["worker_class"]}",  # Use threaded workers (good for I/O-bound model serving)
            "--timeout",
            f"{args["timeout"]}",  # Max time a request can run
            "--backlog",
            f"{args["backlog"]}",  # Max waiting connections
            "--log-level",
            f"{args["log-level"]}",  # Logging verbosity
            "--access-logfile",
            "-",  # Log access to stdout
            "--error-logfile",
            "-",  # Log errors to stderr
        ],
        check=True,
    )

# Or in development mode
else:
    # Initialialize flask app and model
    dev = create_app()

    # Run the flaskapp with the specified
    dev.run(host=args["host"], port=args["port"])
