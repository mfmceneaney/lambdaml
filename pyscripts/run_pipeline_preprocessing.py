import argparse
import json

# Local imports
from lambdaml.preprocess import (
    preprocess_rec_particle,
    label_rec_particle,
    get_kinematics_rec_particle,
)
from lambdaml.pipeline import pipeline_preprocessing
from lambdaml.log import set_global_log_level


# Create argument parser
argparser = argparse.ArgumentParser(description="Run preprocessing pipeline")

# Set available function choices
fn_choices = {
    "preprocess_rec_particle": preprocess_rec_particle,
    "label_rec_particle": label_rec_particle,
    "get_kinematics_rec_particle": get_kinematics_rec_particle,
}

# Add arguments
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
    "--preprocessing_fn",
    type=str,
    default="preprocess_rec_particle",
    help="Preprocessing function to use",
    choices=["preprocess_rec_particle"],
)

argparser.add_argument(
    "--labelling_fn",
    type=str,
    default="label_rec_particle",
    help="Labelling function to use",
    choices=["label_rec_particle"],
)

argparser.add_argument(
    "--kinematics_fn",
    type=str,
    default="get_kinematics_rec_particle",
    help="Kinematics function to use",
    choices=["get_kinematics_rec_particle"],
)

argparser.add_argument(
     "--preprocessing_fn_kwargs",
    type=json.loads,
    default={},
    help="Preprocessing function kwargs dictionary in JSON format, e.g., '{\"a\": 1, \"b\": 2}'"
)

argparser.add_argument(
     "--labelling_fn_kwargs",
    type=json.loads,
    default={},
    help="Labelling function kwargs dictionary in JSON format, e.g., '{\"a\": 1, \"b\": 2}'"
)

argparser.add_argument(
     "--kinematics_fn_kwargs",
    type=json.loads,
    default={},
    help="Kinematics function kwargs dictionary in JSON format, e.g., '{\"a\": 1, \"b\": 2}'"
)

argparser.add_argument(
    "--file_list",
    type=str,
    default=["file_*.hipo"],
    nargs="+"
    help="List of file regexs to use",
)

argparser.add_argument(
    "--banks",
    type=str,
    default=[
        "REC::Particle",
        "REC::Kinematics",
        "MC::Lund",
    ],
    nargs="+",
    help="Banks to use",
)

argparser.add_argument(
    "--step",
    type=int,
    default=1000,
    help="File iteration step size",
)

argparser.add_argument(
    "--out_dataset_path",
    type=str,
    default="/out/src_dataset/",
    help="Directory path for output dataset",
)

argparser.add_argument(
    "--lazy_ds_batch_size",
    type=int,
    default=100000,
    help="Batch size for lazy dataset",
)

argparser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of worker processes for writing dataset",
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

# Loop names of functional arguments and check if a function was actually passed
fn_names = ("preprocessing_fn", "labelling_fn", "kinematics_fn")
for fn_name in fn_names:
    if args[fn_name] is not None and args[fn_name] in fn_choices and callable(fn_choices[args[fn_name]]):

        # Check if any kwargs are given
        if args[fn_name+"_kwargs"] is not None:
            args[fn_name] = lambda *args: fn_choices[args[fn_name]](*args, **args[fn_name+"_kwargs"])
        else:
            args[fn_name] = fn_choices[args[fn_name]]

# Remove config argument
args.pop("config")

# Run pipeline
pipeline_preprocessing(**args)
