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

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="Path to config yaml",
)

argparser.add_argument(
    "--is_tudataset",
    type=bool,
    default=False,
    help="Whether to use tudataset or not",
)

argparser.add_argument(
    "--use_lazy_dataset",
    type=bool,
    default=False,
    help="Whether to use lazy dataset or not",
)

argparser.add_argument(
    "--out_dir",
    type=str,
    default="",
    help="Path to output directory",
)

transforms_dict = {
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
    default="tgt_dataset/",
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
    type=float,
    default=1.0,
    choices=[None, "sigmoid_growth", "sigmoid_decay", "linear_growth", "linear_decay"],
    help="Temperature function",
)

argparser.add_argument(
    "--alpha_fn",
    type=None,
    default=1.0,
    choices=[None, "sigmoid_growth", "sigmoid_decay", "linear_growth", "linear_decay"],
    help="Alpha function",
)

argparser.add_argument(
    "--lambda_fn",
    type=None,
    default="sigmoid_growth",
    choices=[None, "sigmoid_growth", "sigmoid_decay", "linear_growth", "linear_decay"],
    help="Lambda function",
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
    type=tuple,
    default=(i for i in range(3, 11)),
    help="Kinematic indices",
)

argparser.add_argument(
    "--kin_xlabels",
    type=tuple,
    default=(
        "$Q^2$ (GeV$^2$)",
        "$\\nu$",
        "$W$ (GeV)",
        "$x$",
        "$y$",
        "$z_{p\\pi^{-}}$",
        "$x_{F p\\pi^{-}}$",
        "$M_{p\\pi^{-}}$ (GeV)",
    ),
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

argparser.parse_arguments()


pipeline_titok(
    is_tudataset=False,
    use_lazy_dataset=False,
    out_dir="",
    transform=None,  # T.Compose([T.ToUndirected(),T.KNNGraph(k=6),T.NormalizeFeatures()]),
    max_idx=1000,
    ds_split=(0.8, 0.2),
    src_root="src_dataset/",
    tgt_root="tgt_dataset/",
    # loader arguments
    batch_size=32,
    drop_last=True,
    # Model
    device_name=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    nepochs=200,
    num_classes=2,
    gnn_type="gin",
    hdim_gnn=64,
    num_layers_gnn=3,
    dropout_gnn=0.4,
    heads=4,
    num_layers_clf=3,
    hdim_clf=128,
    dropout_clf=0.4,
    # Learning rate arguments
    lr=0.001,
    lr_scheduler_type="linear",  # None/'', step, and linear
    lr_kwargs=None,  # NOTE: default for step
    soft_labels_temp=2,
    confidence_threshold=0.8,
    temp_fn=1.0,
    alpha_fn=1.0,
    lambda_fn=sigmoid_growth,
    coeff_mmd=0.3,
    coeff_auc=0.01,
    coeff_soft=0.25,
    pretrain_frac=0.2,
    verbose=False,
    metrics_plot_path="metrics_plot.pdf",
    metrics_plot_figsize=(24, 12),
    logs_path="logs.json",
    tsne_plot_path="tsne_plot.pdf",
    tsne_plot_figsize=(20, 8),
    # Plot kinematics arguments
    kin_indices=(i for i in range(3, 11)),
    kin_xlabels=(
        "$Q^2$ (GeV$^2$)",
        "$\\nu$",
        "$W$ (GeV)",
        "$x$",
        "$y$",
        "$z_{p\\pi^{-}}$",
        "$x_{F p\\pi^{-}}$",
        "$M_{p\\pi^{-}}$ (GeV)",
    ),
    src_kinematics_plot_path="src_kinematics_plot.pdf",
    tgt_kinematics_plot_path="tgt_kinematics_plot.pdf",
    kinematics_axs=None,
    # Model save arguments
    encoder_path="encoder.pt",
    encoder_params_path="encoder_params.json",
    clf_path="clf.pt",
    clf_params_path="clf_params.json",
    # Optuna trial
    trial=None,
)
