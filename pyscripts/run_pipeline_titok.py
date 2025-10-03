# pylint: disable=no-member
import yaml
import json
import argparse
import torch
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import StepLR

# Local imports
from lambdaml.pipeline import pipeline_titok
from lambdaml.functional import (
    sigmoid_decay,
    sigmoid_growth,
    linear_decay,
    linear_growth,
)
from lambdaml.util import (
    float_or_choice,
    load_yaml,
)
from lambdaml.log import set_global_log_level


argparser = argparse.ArgumentParser(description="Run TIToK training pipeline")

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

argparser.add_argument(
    "--soft_labels_temp",
    type=float,
    default=2,
    help="Temperature for soft labels",
)

argparser.add_argument(
    "--confidence_threshold",
    type=float,
    default=0.8,
    help="Confidence threshold for soft labels",
)

fn_choices = {
    "sigmoid_growth": sigmoid_growth,
    "sigmoid_decay": sigmoid_decay,
    "linear_growth": linear_growth,
    "linear_decay": linear_decay,
}

argparser.add_argument(
    "--temp_fn",
    type=lambda value: float_or_choice(value,choices=["sigmoid_growth", "sigmoid_decay", "linear_growth", "linear_decay"]),
    default=1.0,
    help="Temperature function",
)

argparser.add_argument(
     "--temp_fn_kwargs",
    type=json.loads,
    default=None,
    help="Temperature function kwargs dictionary in JSON format, e.g., '{\"a\": 1, \"b\": 2}'"
)

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
    "--lambda_fn",
    type=lambda value: float_or_choice(value,choices=["sigmoid_growth", "sigmoid_decay", "linear_growth", "linear_decay"]),
    default="sigmoid_growth",
    help="Lambda function",
)

argparser.add_argument(
     "--lambda_fn_kwargs",
    type=json.loads,
    default=None,
    help="Lambda function kwargs dictionary in JSON format, e.g., '{\"a\": 1, \"b\": 2}'"
)

argparser.add_argument(
    "--coeff_mmd",
    type=float,
    default=0.3,
    help="Coefficient for MMD loss",
)

argparser.add_argument(
    "--coeff_auc",
    type=float,
    default=0.01,
    help="Coefficient for AUC loss",
)

argparser.add_argument(
    "--coeff_soft",
    type=float,
    default=0.25,
    help="Coefficient for soft labels loss",
)

argparser.add_argument(
    "--pretrain_frac",
    type=float,
    default=0.2,
    help="Fraction of epochs for which to pretrain before assigning soft labels",
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

# Replace values in args that are aliases for complex classes
args["transform"] = transform_choices[args["transform"]] if args["transform"] is not None else None

# Loop names of functional arguments and check if a function was actually passed
fn_names = ("temp_fn", "alpha_fn", "lambda_fn")
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

# Run pipeline
pipeline_titok(**args)
