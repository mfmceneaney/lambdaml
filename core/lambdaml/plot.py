# PLOT
# pylint: disable=no-member
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE

# Local imports
from .log import setup_logger


# Set module logger
logger = setup_logger(__name__)


# Plot metrics by epoch
def plot_epoch_metrics(
    ax,
    nepochs,
    title="",
    xlabel="",
    ylabel="",
    yscale=None,
    xscale=None,
    legend_bbox_to_anchor=(1.05, 1),
    legend_loc="upper left",
    epoch_metrics=(),
    plot_kwargs=(),
    normalize_to_max=True,
):

    # Check dimensions of metrics and plotting arguments lists
    if len(epoch_metrics) != len(plot_kwargs):
        raise ValueError(
            f"Number of epoch metrics ({len(epoch_metrics)}) does not match number of plot kwargs ({len(plot_kwargs)})"
        )

    # Loop and plot metrics
    for idx, epoch_metric in enumerate(epoch_metrics):
        ax.plot(
            range(nepochs),
            epoch_metric / np.max(epoch_metric) if normalize_to_max else epoch_metric,
            **plot_kwargs[idx],
        )

    # Set up plot
    ax.set_title(title, usetex=True)
    ax.set_xlabel(xlabel, usetex=True)
    ax.set_ylabel(ylabel, usetex=True)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xscale is not None:
        ax.set_xscale(xscale)
    if legend_loc is not None and legend_bbox_to_anchor is None:
        ax.legend(loc=legend_loc)
    if legend_loc is not None and legend_bbox_to_anchor is not None:
        if np.any([el > 1.0 or el < 0.0 for el in legend_bbox_to_anchor]):
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc)


# Plot ROC
def plot_roc(
    ax,
    fpr=(),
    tpr=(),
    auc=0.0,
    best_fpr=0.0,
    best_tpr=0.0,
    best_fom=0.0,
    best_thr=0.0,
):
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
    ax.scatter(
        best_fpr,
        best_tpr,
        color="red",
        marker="*",
        s=100,
        label=f"Max FOM \n(FOM={best_fom:.2f})\n(Thr={best_thr:.2f})",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Classifier ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)


# Plot domain predictions with KS statistic
def plot_domain_preds(ax, src_preds, tgt_preds, bins=50):
    stat, p_value = ks_2samp(src_preds, tgt_preds)
    ax.hist(
        src_preds,
        bins=bins,
        range=(0, 1),
        alpha=0.6,
        label="Source Domain",
        color="skyblue",
        density=True,
    )
    ax.hist(
        tgt_preds,
        bins=bins,
        range=(0, 1),
        alpha=0.6,
        label="Target Domain",
        color="salmon",
        density=True,
    )
    ax.plot([], [], " ", label=f"KS test statistic: {stat:.4f}, p-value: {p_value:.4g}")
    ax.set_xlim([0.0, 1.0])
    ax.set_title("Classifier Output Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)


def collect_embeddings(encoder, clf, loader, device, domain_label):
    encoder.eval()
    clf.eval()
    all_embeds, all_labels, all_domains, all_preds = [], [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x = encoder(data.x, data.edge_index, data.batch)
            logits = clf(x)
            preds = F.softmax(logits, dim=0).argmax(dim=1)
            all_embeds.append(x.cpu())
            all_labels.append(data.y.cpu())
            all_domains.append(
                torch.full((x.size(0),), domain_label)
            )  # 0=source, 1=target
            all_preds.append(preds.cpu())

    return (
        torch.cat(all_embeds, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_domains, dim=0),
        torch.cat(all_preds, dim=0),
    )


def plot_tsne(ax, embeddings, labels, domains, title="t-SNE of Graph Embeddings"):
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)

    embeds_2d = tsne.fit_transform(embeddings)

    for domain in [0, 1]:  # source vs target
        for label in torch.unique(labels):
            idx = (domains == domain) & (labels == label)
            ax.scatter(
                embeds_2d[idx, 0],
                embeds_2d[idx, 1],
                label=f"{'Src' if domain==0 else 'Tgt'} - Class {label.item()}",
                alpha=0.6,
                marker="o" if domain == 0 else "*",
                color="b" if label.item() == 0 else "r",
                s=20,
            )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(title)


def get_kinematics(
    encoder,
    clf,
    dataloader,
    threshold=0.7,
    device="cuda",
    class_idx_signal=1,
    class_idx_background=0,
):
    """
    Plots histograms of each kinematic variable for predicted signal and background.
    """
    encoder.eval()
    clf.eval()
    all_sg_kin = []
    all_bg_kin = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            feats = encoder(data.x, data.edge_index, data.batch)
            logits = clf(feats)
            probs = F.softmax(logits, dim=1)
            pred_probs, pred_classes = probs.max(dim=1)

            # Apply threshold selection
            selected = pred_probs >= threshold
            selected_classes = pred_classes[selected]
            selected_kinematics = data.kinematics[selected]

            for k, cls in zip(selected_kinematics, selected_classes):
                if cls.item() == class_idx_signal:
                    all_sg_kin.append(k.cpu())
                elif cls.item() == class_idx_background:
                    all_bg_kin.append(k.cpu())

    if not all_sg_kin or not all_bg_kin:
        print("Not enough events passed the threshold to plot.")
        return all_sg_kin, all_bg_kin

    # Convert to tensors
    sg_kin = torch.stack(all_sg_kin)  # [n_sg, n_kin]
    bg_kin = torch.stack(all_bg_kin)  # [n_bg, n_kin]

    return sg_kin, bg_kin


def plot_kinematics(
    axs,
    sg_kin,
    bg_kin,
    kin_indices=None,
    kin_xlabels=None,
    sg_hist_kwargs=None,
    bg_hist_kwargs=None,
):

    # Check arguments
    if sg_hist_kwargs is None:
        sg_hist_kwargs = {
            "bins": 50,
            "alpha": 0.6,
            "label": "Signal",
            "color": "C0",
            "density": True,
        }
    if bg_hist_kwargs is None:
        bg_hist_kwargs = {
            "bins": 50,
            "alpha": 0.6,
            "label": "Background",
            "color": "C1",
            "density": True,
        }

    # Set number of kinematics
    n_kin = sg_kin.size(1) if type(sg_kin) == torch.Tensor else 0
    if kin_indices is None:
        kin_indices = [i for i in range(n_kin)]

    if n_kin < len(kin_indices) or len(kin_indices) != len(kin_xlabels):
        raise ValueError(
            "Number of kinematics is not consistent "
            + f"sg_kin.size(1) = {n_kin:d} ,"
            + f"len(kin_indices) = {len(kin_indices):d} ,"
            + f"len(kin_xlabels) = {len(kin_xlabels):d}"
        )

    # Set kinematics labels
    if kin_xlabels is None:
        kin_xlabels = [f"Kin_{i}" for i in kin_indices]

    # Set and flatten axes
    fig = None
    if axs is None or len(axs) == 0:
        fig, axs = plt.subplots(
            nrows=(len(kin_indices) + 1) // 2,
            ncols=2,
            figsize=(14, 4 * ((len(kin_indices) + 1) // 2)),
        )
    axs = axs.flatten()

    # Turn off unused axes
    for idx in range(len(axs) - len(kin_indices)):
        axs[-1 - idx].axis("off")

    # Loop and plot kinematics
    for i, kin_idx in enumerate(kin_indices):
        axs[i].hist(sg_kin[:, kin_idx], **sg_hist_kwargs)
        axs[i].hist(bg_kin[:, kin_idx], **bg_hist_kwargs)
        axs[i].set_xlabel(kin_xlabels[i], usetex=True)
        axs[i].legend()

    return fig, axs
