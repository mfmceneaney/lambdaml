# PIPELINE
# pylint: disable=no-member
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import random_split
import os.path as osp
from os import makedirs
import matplotlib.pyplot as plt
from tqdm import tqdm
import hipopy.hipopy as hp

# Local imports
from .data import (
    SmallDataset,
    LazyDataset,
)
from .preprocess import (
    preprocess_rec_particle,
    label_rec_particle,
    get_kinematics_rec_particle,
    get_bank_keys,
    get_event_table,
)
from .models import FlexibleGNNEncoder, GraphClassifier
from .train import (
    sigmoid_growth,
    train_titok,
)
from .validate import (
    val_titok,
    get_best_threshold,
)
from .plot import (
    plot_epoch_metrics,
    plot_roc,
    plot_domain_preds,
    collect_embeddings,
    plot_tsne,
    get_kinematics,
    plot_kinematics,
)
from .util import save_json
from .log import setup_logger


# Set module logger
logger = setup_logger(__name__)


def pipeline_titok(
    is_tudataset=True,
    use_lazy_dataset=False,
    out_dir="",
    transform=None,  # T.Compose([T.ToUndirected(),T.KNNGraph(k=6),T.NormalizeFeatures()]),
    max_idx=0,
    ds_split=(0.8, 0.2),
    src_root="PROTEINS",
    tgt_root=None,
    # loader arguments
    batch_size=32,
    drop_last=True,
    # Model
    device_name="cuda" if torch.cuda.is_available() else "cpu",
    nepochs=100,
    num_classes=2,
    sg_idx=1,
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
    lr_scheduler_type=None,  # None/'', step, and linear
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
    kin_indices=None,  # (i for i in range(3, 11)),
    kin_xlabels=None,  # ("$Q^2$ (GeV$^2$)","$\\nu$","$W$ (GeV)","$x$","$y$","$z_{p\\pi^{-}}$","$x_{F p\\pi^{-}}$","$M_{p\\pi^{-}}$ (GeV)",),  # 'idxe', 'idxp', 'idxpi',
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
    metric_fn=lambda logs: logs[0]["auc"],  # Available logs are [val_logs]
):

    # Check arguments
    if lr_kwargs is None:
        lr_kwargs = {}

    # Set device
    logger.info("Using device: %s", device_name)
    device = torch.device(device_name)

    # Create output directory
    if out_dir is not None and len(out_dir) > 0:
        makedirs(out_dir, exist_ok=True)

    # Expand paths
    src_root_exp = osp.expanduser(src_root) if src_root is not None else None
    tgt_root_exp = osp.expanduser(tgt_root) if tgt_root is not None else None

    # Load TUDataset or custom dataset
    src_ds, tgt_ds = None, None
    if is_tudataset and src_root is not None:

        # Shuffle and split into two subsets
        if src_root == tgt_root or tgt_root is None or len(tgt_root) == 0:
            logger.info("Loading full TUDataset: %s", src_root_exp)
            full_ds = TUDataset(
                root=osp.dirname(src_root_exp), name=osp.basename(src_root_exp)
            )
            total_len = len(full_ds)
            split_len = total_len // 2
            src_ds, tgt_ds = random_split(full_ds, [split_len, total_len - split_len])

        # Or load two datasets
        else:
            logger.info("Loading source TUDataset: %s", src_root_exp)
            src_ds = TUDataset(
                root=osp.dirname(src_root_exp), name=osp.basename(src_root_exp)
            )
            logger.info("Loading target TUDataset: %s", tgt_root_exp)
            tgt_ds = TUDataset(
                root=osp.dirname(tgt_root_exp), name=osp.basename(tgt_root_exp)
            )

    # Load a custom pyg dataset
    elif src_root is not None and tgt_root is not None:
        if not use_lazy_dataset:
            logger.info("Loading source SmallDataset: %s", src_root_exp)
            src_ds = SmallDataset(
                src_root_exp, transform=transform, pre_transform=None, pre_filter=None
            )
            logger.info("Loading target SmallDataset: %s", tgt_root_exp)
            tgt_ds = SmallDataset(
                tgt_root_exp, transform=transform, pre_transform=None, pre_filter=None
            )

        else:
            logger.info("Loading source LazyDataset: %s", src_root_exp)
            src_ds = LazyDataset(
                src_root_exp, transform=transform, pre_transform=None, pre_filter=None
            )
            logger.info("Loading target LazyDataset: %s", tgt_root_exp)
            tgt_ds = LazyDataset(
                tgt_root_exp, transform=transform, pre_transform=None, pre_filter=None
            )

    # Take subset of datasets
    if max_idx > 0:
        src_ds = src_ds[0:max_idx]
        tgt_ds = tgt_ds[0:max_idx]

    # Split datasets
    logger.info("Splitting datasets with ratio: %s", ds_split)
    src_train_ds, src_val_ds = random_split(src_ds, ds_split)
    tgt_train_ds, tgt_val_ds = random_split(tgt_ds, ds_split)
    logger.debug("len(src_train_ds) = %d", len(src_train_ds))
    logger.debug("len(src_val_ds) = %d", len(src_val_ds))
    logger.debug("len(tgt_train_ds) = %d", len(tgt_train_ds))
    logger.debug("len(tgt_val_ds) = %d", len(tgt_val_ds))
    logger.debug("src_train_ds[0] = %s", src_train_ds[0] if len(src_train_ds) > 0 else None)
    logger.debug("tgt_train_ds[0] = %s", tgt_train_ds[0] if len(tgt_train_ds) > 0 else None)
    logger.debug("src_val_ds[0] = %s", src_val_ds[0] if len(src_val_ds) > 0 else None)
    logger.debug("tgt_val_ds[0] = %s", tgt_val_ds[0] if len(tgt_val_ds) > 0 else None)

    # Create DataLoaders
    logger.info("Creating source training DataLoader")
    src_train_loader = DataLoader(
        src_train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    logger.debug("len(src_train_loader) = %d", len(src_train_loader))

    logger.info("Creating target training DataLoader")
    tgt_train_loader = DataLoader(
        tgt_train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    logger.debug("len(tgt_train_loader) = %d", len(tgt_train_loader))

    # Create DataLoaders
    logger.info("Creating source validation DataLoader")
    src_val_loader = DataLoader(
        src_val_ds, batch_size=batch_size, shuffle=False, drop_last=drop_last
    )
    logger.debug("len(src_train_loader) = %d", len(src_train_loader))
    
    logger.info("Creating target validation DataLoader")
    tgt_val_loader = DataLoader(
        tgt_val_ds, batch_size=batch_size, shuffle=False, drop_last=drop_last
    )
    logger.debug("len(tgt_val_loader) = %d", len(tgt_val_loader))

    # --------------------------------------------------------#
    # Create model
    logger.info("Creating GNN Model")

    num_node_features = src_ds[0].num_node_features

    encoder = FlexibleGNNEncoder(
        in_dim=num_node_features,
        hidden_dim=hdim_gnn,
        num_layers=num_layers_gnn,
        gnn_type=gnn_type,  # Try 'gcn', 'sage', 'gat', 'gin'
        dropout=dropout_gnn,
        heads=heads,  # Only relevant for GAT
    ).to(device)

    clf = GraphClassifier(
        in_dim=hdim_gnn * (heads if gnn_type == "gat" else 1),
        out_dim=num_classes,
        num_layers=num_layers_clf,
        hidden_dim=hdim_clf,
        dropout=dropout_clf,
    ).to(device)

    # ---------- Set optimizer and learning rate scheduler ----------#
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(clf.parameters()), lr=lr
    )
    lr_scheduler = None
    if lr_scheduler_type == "step":
        lr_scheduler = StepLR(optimizer, **lr_kwargs)
    if lr_scheduler_type == "linear":
        lr_lambda = lambda epoch: (1 - (epoch / nepochs))
        lr_scheduler = LambdaLR(optimizer, lr_lambda)

    # ----- Train model
    logger.info("Training model")
    train_logs, soft_labels = train_titok(
        encoder,
        clf,
        optimizer,
        src_train_loader,
        tgt_train_loader,
        src_val_loader,
        tgt_val_loader,
        num_classes=num_classes,
        soft_labels_temp=soft_labels_temp,
        nepochs=nepochs,
        lr_scheduler=lr_scheduler,
        confidence_threshold=confidence_threshold,
        temp_fn=temp_fn,
        alpha_fn=alpha_fn,
        lambda_fn=lambda_fn,
        coeff_mmd=coeff_mmd,
        coeff_auc=coeff_auc,
        coeff_soft=coeff_soft,
        pretrain_frac=pretrain_frac,
        device=device,
        verbose=verbose,
        trial=trial,
        metric_fn=metric_fn,
        sg_idx=sg_idx,
    )
    logger.debug("train_logs = %s", train_logs)
    logger.debug("soft_labels = %s", soft_labels)

    # Save model
    logger.info("Saving GNN Encoder")
    torch.save(encoder.state_dict(), osp.join(out_dir, encoder_path))

    # Save model parameters to json
    logger.info("Saving GNN Encoder Parameters")
    save_json(
        osp.join(out_dir, encoder_params_path),
        {
            "gnn_type": gnn_type,
            "in_dim_gnn": num_node_features,
            "hdim_gnn": hdim_gnn,
            "num_layers_gnn": num_layers_gnn,
            "dropout_gnn": dropout_gnn,
            "heads_gnn": heads,
        },
    )

    # Save classifier
    logger.info("Saving Classifier")
    torch.save(clf.state_dict(), osp.join(out_dir, clf_path))

    # Save classifier parameters to json
    logger.info("Saving Classifier Parameters")
    save_json(
        osp.join(out_dir, clf_params_path),
        {
            "in_dim_clf": hdim_gnn * (heads if gnn_type == "gat" else 1),
            "num_layers_clf": num_layers_clf,
            "hdim_clf": hdim_clf,
            "dropout_clf": dropout_clf,
            "out_dim_clf": num_classes,
        },
    )

    # Record output paths of models and parameters for trial
    if trial is not None:
        logger.info("Setting optuna trial attributes")
        trial.set_user_attr("encoder_path", encoder_path)
        trial.set_user_attr("encoder_params_path", encoder_params_path)
        trial.set_user_attr("clf_path", clf_path)
        trial.set_user_attr("clf_params_path", clf_params_path)

    # ----- Test model
    temp = temp_fn if not callable(temp_fn) else temp_fn(nepochs, nepochs)
    alpha = alpha_fn if not callable(alpha_fn) else alpha_fn(nepochs, nepochs)
    lambd = lambda_fn if not callable(lambda_fn) else lambda_fn(nepochs, nepochs)
    logger.info("Validating model on source validation dataset")
    src_val_logs = val_titok(
        encoder,
        clf,
        optimizer,
        src_val_loader,
        tgt_val_loader,
        soft_labels,
        return_labels=True,
        pretraining=False,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
        temp=temp,
        alpha=alpha,
        lambd=lambd,
        coeff_mmd=coeff_mmd,
        coeff_auc=coeff_auc,
        coeff_soft=coeff_soft,
        device=device,
        sg_idx=sg_idx,
    )
    logger.debug("src_val_logs = %s", src_val_logs)
    logger.info("Validating model on target validation dataset")
    tgt_val_logs = val_titok(
        encoder,
        clf,
        optimizer,
        tgt_val_loader,
        tgt_val_loader,
        soft_labels,
        return_labels=True,
        pretraining=False,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
        temp=temp,
        alpha=alpha,
        lambd=lambd,
        coeff_mmd=coeff_mmd,
        coeff_auc=coeff_auc,
        coeff_soft=coeff_soft,
        device=device,
        sg_idx=sg_idx,
    )
    logger.debug("tgt_val_logs = %s", tgt_val_logs)

    # Save training logs to json
    logger.info("Saving training logs")
    save_json(
        osp.join(out_dir, logs_path),
        {
            "train": train_logs,
            "src_val": [
                el if not isinstance(el, torch.Tensor) else el.tolist() for el in src_val_logs
            ],
            "tgt_val": [
                el if not isinstance(el, torch.Tensor) else el.tolist() for el in tgt_val_logs
            ],
        },
    )

    # Pop src validation log values
    src_probs = src_val_logs["probs"]
    src_preds = src_val_logs["preds"]
    src_labels = src_val_logs["labels"]

    # Pop tgt validation log values
    tgt_probs = tgt_val_logs["probs"]
    tgt_preds = tgt_val_logs["preds"]

    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=metrics_plot_figsize)

    # Plot loss coefficients
    alpha_values = [
        alpha_fn(e, nepochs) if callable(alpha_fn) else alpha_fn for e in range(nepochs)
    ]
    lambda_values = [
        lambda_fn(e, nepochs) if callable(lambda_fn) else lambda_fn
        for e in range(nepochs)
    ]
    loss_coeffs = [alpha_values, lambda_values]
    loss_coeffs_kwargs = [{"label": "alpha"}, {"label": "lambda"}]
    logger.info("Plotting loss coefficients")
    plot_epoch_metrics(
        axs[0, 1],
        nepochs,
        title="Loss Coefficients",
        xlabel="Epoch",
        ylabel="Loss Coefficient",
        yscale=None,
        xscale=None,
        legend_bbox_to_anchor=None,
        legend_loc="best",
        epoch_metrics=loss_coeffs,
        plot_kwargs=loss_coeffs_kwargs,
        normalize_to_max=False,
    )

    # Plot learning rate
    lrs = [train_logs["lrs"]]
    lrs_kwargs = [{"label": "lr"}]
    logger.info("Plotting learning rates")
    plot_epoch_metrics(
        axs[1, 1],
        nepochs,
        title="Learning Rate",
        xlabel="Epoch",
        ylabel="Learning Rate",
        yscale="log",
        xscale=None,
        legend_bbox_to_anchor=None,
        legend_loc="best",
        epoch_metrics=lrs,
        plot_kwargs=lrs_kwargs,
        normalize_to_max=False,
    )

    # Plot training and validation losses
    train_losses = [train_logs[key] for key in train_logs if "train_loss" in key]
    val_losses = [train_logs[key] for key in train_logs if "val_loss" in key]
    losses = [*train_losses, *val_losses]
    train_losses_kwargs = [{"label": key} for key in train_logs if "train_loss" in key]
    val_losses_kwargs = [
        {"label": key, "linestyle": ":"} for key in train_logs if "val_loss" in key
    ]
    losses_kwargs = [*train_losses_kwargs, *val_losses_kwargs]
    logger.info("Plotting losses")
    plot_epoch_metrics(
        axs[0, 2],
        nepochs,
        title="Losses",
        xlabel="Epoch",
        ylabel="Loss",
        yscale="log",
        xscale=None,
        legend_bbox_to_anchor=(1.05, 1),
        legend_loc="upper left",
        epoch_metrics=losses,
        plot_kwargs=losses_kwargs,
        normalize_to_max=True,
    )

    # Plot training and validation accuracies
    train_accs = [train_logs[key] for key in train_logs if "train_acc" in key]
    val_accs = [train_logs[key] for key in train_logs if "val_acc" in key]
    accs = [*train_accs, *val_accs]
    train_accs_kwargs = [{"label": key} for key in train_logs if "train_acc" in key]
    val_accs_kwargs = [
        {"label": key, "linestyle": ":"} for key in train_logs if "val_acc" in key
    ]
    accs_kwargs = [*train_accs_kwargs, *val_accs_kwargs]
    logger.info("Plotting accuracies")
    plot_epoch_metrics(
        axs[1, 2],
        nepochs,
        title="Accuracies",
        xlabel="Epoch",
        ylabel="Accuracy",
        yscale=None,
        xscale=None,
        legend_bbox_to_anchor=(1.05, 1),
        legend_loc="upper left",
        epoch_metrics=accs,
        plot_kwargs=accs_kwargs,
        normalize_to_max=True,
    )

    # Plot domain predictions
    logger.info("Plotting domain predictions")
    plot_domain_preds(axs[0, 0], src_probs[:, 1], tgt_probs[:, 1])

    # Plot ROC AUC curve
    roc_info, _ = get_best_threshold(src_labels, src_probs[:, 1])
    best_thr = roc_info["best_thr"]
    logger.info("Plotting ROC curve")
    plot_roc(axs[1, 0], **roc_info)

    # Save and show plot
    logger.info("Saving training metrics figure")
    plt.tight_layout()
    fig.savefig(osp.join(out_dir, metrics_plot_path), bbox_inches="tight")

    # ----- t-SNE model representation
    logger.info("Collecting embeddings")
    src_embeds, src_labels, src_domains, src_preds = collect_embeddings(
        encoder, clf, src_val_loader, device, domain_label=0
    )
    tgt_embeds, _, tgt_domains, tgt_preds = collect_embeddings(
        encoder, clf, tgt_val_loader, device, domain_label=1
    )

    # Combine
    all_embeds = torch.cat([src_embeds, tgt_embeds], dim=0)
    all_domains = torch.cat([src_domains, tgt_domains], dim=0)
    all_preds = torch.cat([src_preds, tgt_preds], dim=0)
    labels_and_preds = torch.cat([src_labels, tgt_preds], dim=0)

    # Create figure
    logger.info("Plotting t-SNE representations")
    fig, axs = plt.subplots(1, 2, figsize=tsne_plot_figsize)

    # Plot
    plot_tsne(
        axs[0],
        all_embeds.numpy(),
        labels_and_preds,
        all_domains,
        title="t-SNE representation : true source labels",
    )
    plot_tsne(
        axs[1],
        all_embeds.numpy(),
        all_preds,
        all_domains,
        title="t-SNE representation : model predicted labels",
    )

    # Save and show t-SNE fig
    logger.info("Saving t-SNE representation figure")
    plt.tight_layout()
    fig.savefig(osp.join(out_dir, tsne_plot_path))

    # Check if arguments have been supplied
    if (
        kin_indices is not None
        and kin_xlabels is not None
        and len(kin_indices) == len(kin_xlabels)
    ):

        # Get kinematics for source and target domains
        logger.info("Collecting kinematics arrays")
        src_sg_kin, src_bg_kin = get_kinematics(
            encoder,
            clf,
            src_val_loader,
            threshold=best_thr,
            device=device,
            class_idx_signal=1,
            class_idx_background=0,
        )
        tgt_sg_kin, tgt_bg_kin = get_kinematics(
            encoder,
            clf,
            tgt_val_loader,
            threshold=best_thr,
            device=device,
            class_idx_signal=1,
            class_idx_background=0,
        )

        try:

            # Plot kinematics for source and target domains
            logger.info("Plotting kinematics distributions")
            src_fig, _ = plot_kinematics(
                kinematics_axs,
                src_sg_kin,
                src_bg_kin,
                kin_indices=kin_indices,
                kin_xlabels=kin_xlabels,
                sg_hist_kwargs={
                    "bins": 50,
                    "alpha": 0.6,
                    "label": "Signal",
                    "color": "C0",
                    "density": True,
                },
                bg_hist_kwargs={
                    "bins": 50,
                    "alpha": 0.6,
                    "label": "Background",
                    "color": "C1",
                    "density": True,
                },
            )
            tgt_fig, _ = plot_kinematics(
                kinematics_axs,
                tgt_sg_kin,
                tgt_bg_kin,
                kin_indices=kin_indices,
                kin_xlabels=kin_xlabels,
                sg_hist_kwargs={
                    "bins": 50,
                    "alpha": 0.6,
                    "label": "Signal",
                    "color": "C0",
                    "density": True,
                },
                bg_hist_kwargs={
                    "bins": 50,
                    "alpha": 0.6,
                    "label": "Background",
                    "color": "C1",
                    "density": True,
                },
            )

            # Save and plot kinematics figures
            logger.info("Saving kinematics figure")
            plt.tight_layout()
            src_fig.savefig(osp.join(out_dir, src_kinematics_plot_path))
            tgt_fig.savefig(osp.join(out_dir, tgt_kinematics_plot_path))

        except ValueError:
            pass

    # Set output paths
    paths = [
        metrics_plot_path,
        tsne_plot_path,
        src_kinematics_plot_path,
        tgt_kinematics_plot_path,
    ]
    paths = [osp.join(out_dir, path) for path in paths]

    return roc_info, src_val_logs, tgt_val_logs, paths


def pipeline_preprocessing(
    # Set functions and kwargs
    preprocessing_fn=preprocess_rec_particle,
    labelling_fn=label_rec_particle,
    kinematics_fn=get_kinematics_rec_particle,
    preprocessing_fn_kwargs=None,
    labelling_fn_kwargs=None,
    kinematics_fn_kwargs=None,
    # Set input files, banks, and step size
    file_list=("file_*.hipo"),
    banks=(
        "REC::Particle",
        "REC::Kinematics",
        "MC::Lund",
    ),
    step=1000,
    # Set output dataset and path names
    out_dataset_path="src_root/",
    lazy_ds_batch_size=100000,
    num_workers=0,
):

    # Check arguments
    if preprocessing_fn_kwargs is None:
        preprocessing_fn_kwargs = {}
    if labelling_fn_kwargs is None:
        labelling_fn_kwargs = {}
    if kinematics_fn_kwargs is None:
        kinematics_fn_kwargs = {}

    # Iterate hipo files
    logger.info("Looping input hipo files")
    logger.debug("file_list = %s", file_list)
    logger.debug("banks = %s", banks)
    for batch in tqdm(hp.iterate(file_list, banks=banks, step=step)):

        # Initialize graph list
        datalist = []

        # Set bank names and entry names to look at
        all_keys = list(batch.keys())
        bank_keys = {bank: get_bank_keys(bank, all_keys) for bank in banks}

        # Loop events in batch
        for event_num, _ in enumerate(range(0, len(batch[list(batch.keys())[0]]))):

            # Get data tables
            data_event_tables = {
                bank: get_event_table(bank_keys[bank], event_num, batch, dtype=float)
                for bank in banks
            }

            # Preprocess graph
            x, edge_index = None, None
            if callable(preprocessing_fn):
                x, edge_index = preprocessing_fn(
                    data_event_tables, **preprocessing_fn_kwargs
                )
            else:
                raise TypeError(
                    "Preprocessing function must be calllable, but found type(preprocessing_fn) =",
                    type(preprocessing_fn),
                )

            # Create data from x, edge_index
            data = Data(x=x, edge_index=edge_index)

            # Label graph
            if labelling_fn is not None and callable(labelling_fn):
                data.y = labelling_fn(data_event_tables, **labelling_fn_kwargs)

            # Add kinematics to graph
            if kinematics_fn is not None and callable(kinematics_fn):
                data.kinematics = kinematics_fn(
                    data_event_tables, **kinematics_fn_kwargs
                )

            # Add graph to dataset
            datalist.append(data)

        # Create (or add to) a PyG Dataset
        LazyDataset(
            out_dataset_path,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            datalist=datalist,
            batch_size=lazy_ds_batch_size,
            num_workers=num_workers,
        )
