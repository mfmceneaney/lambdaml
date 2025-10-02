# pylint: disable=no-member
import yaml
import json
import argparse
import torch
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import StepLR
import os

# Local imports
from lambdaml.pipeline import pipeline_da
from lambdaml.functional import (
    sigmoid_decay,
    sigmoid_growth,
    linear_decay,
    linear_growth,
)
from lambdaml.util import float_or_choice
from lambdaml.log import set_global_log_level
from lambdaml.optimize import (
    parse_suggestion_rule,
    optimize,
)


argparser = argparse.ArgumentParser(description="Run hyperparameter optimization on the domain-adversarial pipeline")

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
    "--is_tudataset",
    action="store_true",
    help="Option to use TUDataset datasets (only for non-lazy datasets",
)

argparser.add_argument(
    "--use_lazy_dataset",
    action="store_true",
    help="Option to use lazy datasets (only for non-tudataset datasets)",
)

argparser.add_argument(
    "--out_dir",
    type=str,
    default="experiments/",
    help="Path to output directory",
)

transform_choices = {
    "knn_graph1": T.KNNGraph(k=1),
    "knn_graph2": T.KNNGraph(k=2),
    "knn_graph3": T.KNNGraph(k=3),
    "knn_graph4": T.KNNGraph(k=4),
    "knn_graph5": T.KNNGraph(k=5),
    "knn_graph6": T.KNNGraph(k=6),
    "normalize_features": T.NormalizeFeatures(),
}
argparser.add_argument(
    "--transform",
    type=str,
    default=None,
    choices=[None, "knn_graph1", "knn_graph2", "knn_graph3", "knn_graph4", "knn_graph5", "knn_graph6", "normalize_features"],
    action="append",
    help="Transform(s) to use",
)

argparser.add_argument(
    "--max_idx",
    type=int,
    default=1000,
    help="Maximum index to use in datasets",
)

argparser.add_argument(
    "--ds_split",
    type=tuple,
    default=(0.8, 0.2),
    help="Train and validation split",
)

argparser.add_argument(
    "--src_root",
    type=str,
    default="src_dataset/",
    help="Path to source dataset",
)

argparser.add_argument(
    "--tgt_root",
    type=str,
    default=None,
    help="Path to target dataset",
)

argparser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size",
)

argparser.add_argument(
    "--drop_last",
    type=bool,
    default=True,
    help="Whether to drop last batch or not",
)

argparser.add_argument(
    "--device_name",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device name",
)

argparser.add_argument(
    "--nepochs",
    type=int,
    default=200,
    help="Number of epochs",
)

argparser.add_argument(
    "--num_classes",
    type=int,
    default=2,
    help="Number of classes",
)

argparser.add_argument(
    "--sg_idx",
    type=int,
    default=1,
    help="Index of signal class (must be in [0, num_classes-1])",
)

argparser.add_argument(
    "--gnn_type",
    type=str,
    default="gin",
    choices=["gin", "gcn", "gat", "sage"],
    help="GNN type",
)

argparser.add_argument(
    "--hdim_gnn",
    type=int,
    default=64,
    help="GNN hidden dimension",
)

argparser.add_argument(
    "--num_layers_gnn",
    type=int,
    default=3,
    help="Number of GNN layers",
)

argparser.add_argument(
    "--dropout_gnn",
    type=float,
    default=0.4,
    help="GNN dropout",
)

argparser.add_argument(
    "--heads",
    type=int,
    default=4,
    help="Number of heads in GNN for GAT",
)

argparser.add_argument(
    "--num_layers_clf",
    type=int,
    default=3,
    help="Number of classifier layers",
)

argparser.add_argument(
    "--hdim_clf",
    type=int,
    default=128,
    help="Classifier hidden dimension",
)

argparser.add_argument(
    "--dropout_clf",
    type=float,
    default=0.4,
    help="Classifier dropout",
)

argparser.add_argument(
    "--num_layers_disc",
    type=int,
    default=3,
    help="Number of discriminator layers",
)

argparser.add_argument(
    "--hdim_disc",
    type=int,
    default=128,
    help="Discriminator hidden dimension",
)

argparser.add_argument(
    "--dropout_disc",
    type=float,
    default=0.4,
    help="Discriminator dropout",
)

argparser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="Learning rate",
)

argparser.add_argument(
    "--lr_scheduler_type",
    type=str,
    default=None,
    choices=[None, "linear", "step"],
    help="Learning rate scheduler type",
)

argparser.add_argument(
     "--lr_kwargs",
    type=json.loads,
    default={"step_size": 50, "gamma": 0.5},
    help="Learning rate scheduler kwargs dictionary in JSON format, e.g., '{\"a\": 1, \"b\": 2}'"
)

fn_choices = {
    "sigmoid_growth": sigmoid_growth,
    "sigmoid_decay": sigmoid_decay,
    "linear_growth": linear_growth,
    "linear_decay": linear_decay,
}

argparser.add_argument(
    "--alpha_fn",
    type=lambda value: float_or_choice(value,choices=["sigmoid_growth", "sigmoid_decay", "linear_growth", "linear_decay"]),
    default=1.0,
    help="Alpha function",
)

argparser.add_argument(
     "--alpha_fn_kwargs",
    type=json.loads,
    default=None,
    help="Alpha function kwargs dictionary in JSON format, e.g., '{\"a\": 1, \"b\": 2}'"
)

argparser.add_argument(
    "--verbose",
    type=bool,
    default=False,
    help="Whether to print verbose output or not",
)

argparser.add_argument(
    "--metrics_plot_path",
    type=str,
    default="metrics_plot.pdf",
    help="Path to metrics plot",
)

argparser.add_argument(
    "--metrics_plot_figsize",
    type=tuple,
    default=(24, 12),
    help="Figure size for metrics plot",
)

argparser.add_argument(
    "--logs_path",
    type=str,
    default="logs.json",
    help="Path to logs",
)

argparser.add_argument(
    "--tsne_plot_path",
    type=str,
    default="tsne_plot.pdf",
    help="Path to t-SNE plot",
)

argparser.add_argument(
    "--tsne_plot_figsize",
    type=tuple,
    default=(20, 8),
    help="Figure size for t-SNE plot",
)

argparser.add_argument(
    "--kin_indices",
    type=list,
    default=None,
    help="Kinematic indices",
)

argparser.add_argument(
    "--kin_xlabels",
    type=list,
    default=None,
    help="Kinematic xlabels",
)

argparser.add_argument(
    "--src_kinematics_plot_path",
    type=str,
    default="src_kinematics_plot.pdf",
    help="Path to source kinematics plot",
)

argparser.add_argument(
    "--tgt_kinematics_plot_path",
    type=str,
    default="tgt_kinematics_plot.pdf",
    help="Path to target kinematics plot",
)

argparser.add_argument(
    "--encoder_path",
    type=str,
    default="encoder.pt",
    help="Path for encoder",
)

argparser.add_argument(
    "--encoder_params_path",
    type=str,
    default="encoder_params.json",
    help="Path for encoder params",
)

argparser.add_argument(
    "--clf_path",
    type=str,
    default="clf.pt",
    help="Path for classifier",
)

argparser.add_argument(
    "--clf_params_path",
    type=str,
    default="clf_params.json",
    help="Path for classifier params",
)

argparser.add_argument(
    "--disc_path",
    type=str,
    default="disc.pt",
    help="Path for discriminator",
)

argparser.add_argument(
    "--disc_params_path",
    type=str,
    default="disc_params.json",
    help="Path for discriminator params",
)

argparser.add_argument(
    "--opt__storage_url",
    type=str,
    default="sqlite:///optuna_study.db",
    help="Storage URL for Optuna",
)

argparser.add_argument(
    "--opt__optuna_study_direction",
    type=str,
    default="maximize",
    choices=["minimize", "maximize"],
    help="Optuna study direction",
)

argparser.add_argument(
    "--opt__optuna_study_name",
    type=str,
    default="model_hpo",
    help="Optuna study name",
)

argparser.add_argument(
    "--opt__metric_fn",
    type=str,
    default="auc",
    choices=[
        "auc",
        "best_fpr",
        "best_tpr",
        "best_fom",
        "best_thr",
        "loss",
        "loss_cls",
        "loss_mmd",
        "loss_auc",
        "loss_soft",
        "acc_raw",
        "acc_per_class",
        "acc_balanced",
    ],
    help="Metric name to optimize",
)

argparser.add_argument(
    "--opt__suggestion_rules",
    type=str,
    nargs="*",
    default=[],
    help="Suggestion rules for Optuna, e.g., 'key=int:min:max', 'key=float:min:max[:log][:step]' or 'key=cat:val1,val2,val3'",
)

argparser.add_argument(
    "--opt__pipeline",
    type=str,
    default="pipeline_da",
    choices=["pipeline_da"],
    help="Pipeline to use for optimization",
)
pipeline_choices = {
    "pipeline_da": pipeline_da,
}

argparser.add_argument(
    "--opt__n_trials",
    type=int,
    default=100,
    help="Number of trials for Optuna",
)   

argparser.add_argument(
    "--opt__sampler_name",
    type=str,
    default="tpe",
    choices=["grid", "random", "tpe", "cmaes", "gp", "partialfixed", "nsga2", "qmc"],
    help="Optuna sampler name",
)

argparser.add_argument(
    "--opt__sampler_args",
    type=json.loads,
    default=None,
    help="Optuna sampler args list in JSON format, e.g., '[arg1, arg2]'",
)

argparser.add_argument(
    "--opt__sampler_kwargs",
    type=json.loads,
    default=None,
    help="Optuna sampler kwargs dict in JSON format, e.g., '{\"arg1\": val1, \"arg2\": val2}'",
)

argparser.add_argument(
    "--opt__pruner_name",
    type=str,
    default="median",
    choices=["median", "noprune", "patient", "percentile", "successivehalving", "hyperband", "threshold", "wilcoxon"],
    help="Optuna pruner name",
)

argparser.add_argument(
    "--opt__pruner_args",
    type=json.loads,
    default=None,
    help="Optuna pruner args list in JSON format, e.g., '[arg1, arg2]'",
)

argparser.add_argument(
    "--opt__pruner_kwargs",
    type=json.loads,
    default=None,
    help="Optuna pruner kwargs dict in JSON format, e.g., '{\"arg1\": val1, \"arg2\": val2}'",
)

argparser.add_argument(
    "--wandb_mode",
    type=str,
    default="offline",
    help="WANDB_MODE environment variable for wandb project mode",
)

argparser.add_argument(
    "--wandb_dir",
    type=str,
    default="wandb_logs",
    help="WANDB_DIR environment variable for path wandb logs directory",
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

# Pop wandb specific arguments and set as environment variables
args_keys = list(args.keys())
for key in args_keys:
    if key.startswith("wandb"):
        val = args.pop(key)
        if key=='wandb_dir' and not val.startswith(osp.sep):
            val = osp.abspath(osp.join(out_dir, val))
        os.environ[key.upper()] = val

# Pop optuna specific arguments removing "opt__" prefix
opt_args = {}
args_keys = list(args.keys())
for key in args_keys:
    if key.startswith("opt__"):
        opt_args["".join(key.split("opt__")[1])] = args.pop(key)

# Parse suggestion rules
suggestion_rules = {}
if opt_args["suggestion_rules"] is not None:
    for rule_str in opt_args["suggestion_rules"]:
        rule = parse_suggestion_rule(rule_str)
        suggestion_rules.update(rule)
opt_args["suggestion_rules"] = suggestion_rules

# Parse pipeline choice
opt_args["pipeline"] = pipeline_choices[opt_args["pipeline"]]

# Parse metric function choice
metric_fn_name = opt_args["metric_fn"]
opt_args["metric_fn"] = lambda logs: logs[0][metric_fn_name]

# Replace values in args that are aliases for complex classes
args["transform"] = transform_choices[args["transform"]] if args["transform"] is not None else None

# Loop names of functional arguments and check if a function was actually passed
fn_names = ("alpha_fn",)
for fn_name in fn_names:
    if args[fn_name] is not None and args[fn_name] in fn_choices and callable(fn_choices[args[fn_name]]):

        # Check if any kwargs are given
        if args[fn_name+"_kwargs"] is not None:
            args[fn_name] = lambda *args: fn_choices[args[fn_name]](*args, **args[fn_name+"_kwargs"])
        else:
            args[fn_name] = fn_choices[args[fn_name]]

    # Remove kwargs argument
    args.pop(fn_name+"_kwargs")

# Remove config argument
args.pop("config")

# Add remaining args to opt_args as pipeline_kwargs
opt_args["pipeline_kwargs"] = args

# Run optimization
optimize(**opt_args)
