# EVAL
# pylint: disable=no-member
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

# Local imports
from .functional import (
    loss_da,
    loss_can,
    loss_titok,
)
from .log import setup_logger


# Set module logger
logger = setup_logger(__name__)


def val_da(
    encoder,
    clf,
    disc,
    optimizer,
    src_val_loader,
    tgt_val_loader,
    return_labels=True,
    num_classes=2,
    sg_idx=1,
    alpha=0.5,
    device="cuda:0",
):

    # Set models in eval mode
    encoder.eval()
    clf.eval()
    disc.eval()

    # Initialize variables
    total_loss = 0
    total_loss_cls = 0
    total_loss_disc = 0
    correct = 0
    total = 0
    all_src_probs = []
    all_src_preds = []
    all_src_labels = []
    correct_per_class = torch.zeros(num_classes).to(device)
    total_per_class = torch.zeros(num_classes).to(device)

    # Iterate over source and target loaders in parallel
    with torch.no_grad():
        for src_batch, tgt_batch in zip(src_val_loader, tgt_val_loader):

            # Reset gradients
            optimizer.zero_grad()

            # Source graph forward pass
            logger.debug("src_batch = %s", src_batch)
            src_batch = src_batch.to(device)
            src_feats = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_logits = clf(src_feats)
            src_probs = F.softmax(src_logits, dim=1)
            src_preds = src_probs.argmax(dim=1)
            src_labels = src_batch.y

            # Target graph forward pass
            logger.debug("tgt_batch = %s", tgt_batch)
            tgt_batch = tgt_batch.to(device)
            tgt_feats = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)

            # Source + Target forward pass on discriminator
            dom_feats = torch.cat([src_feats, tgt_feats], dim=0)
            dom_logits = disc(dom_feats, alpha)
            dom_labels = torch.cat(
                [
                    torch.zeros(src_feats.size(0), dtype=torch.long),
                    torch.ones(tgt_feats.size(0), dtype=torch.long),
                ],
                dim=0,
            ).to(device)

            # Compute loss
            loss, loss_cls, loss_disc = loss_da(
                src_logits,
                src_labels,
                dom_logits,
                dom_labels,
            )
            logger.debug("loss_da = %s", loss)

            # Pop losses
            total_loss += loss.item()
            total_loss_cls += loss_cls.item()
            total_loss_disc += loss_disc.item()

            # Compute ROC curve and AUC
            roc_info, _ = get_best_threshold(
                src_labels, src_probs[:, sg_idx], return_arrays=False
            )
            logger.debug("roc_info = %s", roc_info)

            # Count correct predictions
            correct += (src_preds == src_labels).sum().item()
            total += src_labels.size(0)
            if return_labels:
                all_src_probs.extend(src_probs.cpu().tolist())
                all_src_preds.extend(src_preds.cpu().tolist())
                all_src_labels.extend(src_labels.cpu().tolist())

            for i in range(len(src_preds)):
                label = src_labels[i]
                total_per_class[label] += 1
                if src_preds[i] == label:
                    correct_per_class[label] += 1

    # Compute per-class accuracies, avoiding division by zero
    acc_per_class = correct_per_class / (total_per_class + 1e-8)

    # Compute average per-class accuracy
    valid_class_mask = total_per_class > 0
    acc_balanced = acc_per_class[valid_class_mask].mean().item()

    # Compute raw accuracy
    acc_raw = correct / total

    # Convert lists to torch tensors
    all_src_probs = torch.tensor(all_src_probs)
    all_src_preds = torch.tensor(all_src_preds)
    all_src_labels = torch.tensor(all_src_labels)

    logs = {
        **roc_info,
        "loss": total_loss,
        "loss_cls": total_loss_cls,
        "loss_disc": total_loss_disc,
        "acc_raw": acc_raw,
        "acc_per_class": acc_per_class.cpu().tolist(),
        "acc_balanced": acc_balanced,
        "probs": all_src_probs,
        "preds": all_src_preds,
        "labels": all_src_labels,
    }

    return logs


def val_can(
    encoder,
    clf,
    projector,
    optimizer,
    src_val_loader,
    tgt_val_loader,
    return_labels=True,
    num_classes=2,
    sg_idx=1,
    temp=2.0,
    alpha=0.5,
    device="cuda:0",
):

    # Set models in eval mode
    encoder.eval()
    clf.eval()
    projector.eval()

    # Initialize variables
    total_loss = 0
    total_loss_cls = 0
    total_loss_con = 0
    correct = 0
    total = 0
    all_src_probs = []
    all_src_preds = []
    all_src_labels = []
    correct_per_class = torch.zeros(num_classes).to(device)
    total_per_class = torch.zeros(num_classes).to(device)

    # Iterate over source and target loaders in parallel
    with torch.no_grad():
        for src_batch, tgt_batch in zip(src_val_loader, tgt_val_loader):

            # Reset gradients
            optimizer.zero_grad()

            # Source graph forward pass
            logger.debug("src_batch = %s", src_batch)
            src_batch = src_batch.to(device)
            src_feats = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_logits = clf(src_feats)
            src_probs = F.softmax(src_logits, dim=1)
            src_preds = src_probs.argmax(dim=1)
            src_projs = projector(src_feats)
            src_labels = src_batch.y

            # Target graph forward pass
            logger.debug("tgt_batch = %s", tgt_batch)
            tgt_batch = tgt_batch.to(device)
            tgt_feats = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)
            tgt_projs = projector(tgt_feats)

            # Compute loss
            loss, loss_cls, loss_con = loss_can(
                src_logits,
                src_labels,
                src_projs,
                tgt_projs,
                alpha=alpha,
                temp=temp,
            )
            logger.debug("loss_can = %s", loss)

            # Pop losses
            total_loss += loss.item()
            total_loss_cls += loss_cls.item()
            total_loss_con += loss_con.item()

            # Compute ROC curve and AUC
            roc_info, _ = get_best_threshold(
                src_labels, src_probs[:, sg_idx], return_arrays=False
            )
            logger.debug("roc_info = %s", roc_info)

            # Count correct predictions
            correct += (src_preds == src_labels).sum().item()
            total += src_labels.size(0)
            if return_labels:
                all_src_probs.extend(src_probs.cpu().tolist())
                all_src_preds.extend(src_preds.cpu().tolist())
                all_src_labels.extend(src_labels.cpu().tolist())

            for i in range(len(src_preds)):
                label = src_labels[i]
                total_per_class[label] += 1
                if src_preds[i] == label:
                    correct_per_class[label] += 1

    # Compute per-class accuracies, avoiding division by zero
    acc_per_class = correct_per_class / (total_per_class + 1e-8)

    # Compute average per-class accuracy
    valid_class_mask = total_per_class > 0
    acc_balanced = acc_per_class[valid_class_mask].mean().item()

    # Compute raw accuracy
    acc_raw = correct / total

    # Convert lists to torch tensors
    all_src_probs = torch.tensor(all_src_probs)
    all_src_preds = torch.tensor(all_src_preds)
    all_src_labels = torch.tensor(all_src_labels)

    logs = {
        **roc_info,
        "loss": total_loss,
        "loss_cls": total_loss_cls,
        "loss_con": total_loss_con,
        "acc_raw": acc_raw,
        "acc_per_class": acc_per_class.cpu().tolist(),
        "acc_balanced": acc_balanced,
        "probs": all_src_probs,
        "preds": all_src_preds,
        "labels": all_src_labels,
    }

    return logs


def val_titok(
    encoder,
    clf,
    optimizer,
    src_val_loader,
    tgt_val_loader,
    soft_labels,
    pretraining=False,
    return_labels=True,
    num_classes=2,
    sg_idx=1,
    confidence_threshold=0.8,
    temp=2.0,
    alpha=0.5,
    lambd=1.0,
    coeff_mmd=0.3,
    coeff_auc=0.01,
    coeff_soft=0.25,
    device="cuda:0",
):

    # Set models in eval mode
    encoder.eval()
    clf.eval()

    # Initialize variables
    total_loss = 0
    total_loss_cls = 0
    total_loss_auc = 0
    total_loss_mmd = 0
    total_loss_soft = 0
    correct = 0
    total = 0
    all_src_probs = []
    all_src_preds = []
    all_src_labels = []
    correct_per_class = torch.zeros(num_classes).to(device)
    total_per_class = torch.zeros(num_classes).to(device)

    # Iterate over source and target loaders in parallel
    with torch.no_grad():
        for src_batch, tgt_batch in zip(src_val_loader, tgt_val_loader):

            # Reset gradients
            optimizer.zero_grad()

            # Source graph forward pass
            logger.debug("src_batch = %s", src_batch)
            src_batch = src_batch.to(device)
            src_feats = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_logits = clf(src_feats)
            src_probs = F.softmax(src_logits, dim=1)
            src_preds = src_probs.argmax(dim=1)
            src_labels = src_batch.y

            # Target graph forward pass
            logger.debug("tgt_batch = %s", tgt_batch)
            tgt_batch = tgt_batch.to(device)
            tgt_feats = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)
            tgt_logits = clf(tgt_feats)

            # Compute loss
            loss, loss_cls, loss_mmd, loss_auc, loss_soft = loss_titok(
                src_feats,
                src_logits,
                src_labels,
                tgt_feats,
                tgt_logits,
                soft_labels,
                loss_auc_alpha=alpha,
                loss_soft_temperature=temp,
                confidence_threshold=confidence_threshold,
                coeff_mmd=coeff_mmd,
                lambd=lambd,
                coeff_auc=coeff_auc,
                coeff_soft=coeff_soft,
                num_classes=num_classes,
                pretraining=pretraining,
                device=device,
            )
            logger.debug("loss_titok = %s", loss)

            # Pop losses
            total_loss += loss.item()
            total_loss_cls += loss_cls.item()
            total_loss_mmd += loss_mmd.item()
            total_loss_auc += loss_auc.item()
            total_loss_soft += loss_soft.item()

            # Compute ROC curve and AUC
            roc_info, _ = get_best_threshold(
                src_labels, src_probs[:, sg_idx], return_arrays=False
            )
            logger.debug("roc_info = %s", roc_info)

            # Count correct predictions
            correct += (src_preds == src_labels).sum().item()
            total += src_labels.size(0)
            if return_labels:
                all_src_probs.extend(src_probs.cpu().tolist())
                all_src_preds.extend(src_preds.cpu().tolist())
                all_src_labels.extend(src_labels.cpu().tolist())

            for i in range(len(src_preds)):
                label = src_labels[i]
                total_per_class[label] += 1
                if src_preds[i] == label:
                    correct_per_class[label] += 1

    # Compute per-class accuracies, avoiding division by zero
    acc_per_class = correct_per_class / (total_per_class + 1e-8)

    # Compute average per-class accuracy
    valid_class_mask = total_per_class > 0
    acc_balanced = acc_per_class[valid_class_mask].mean().item()

    # Compute raw accuracy
    acc_raw = correct / total

    # Convert lists to torch tensors
    all_src_probs = torch.tensor(all_src_probs)
    all_src_preds = torch.tensor(all_src_preds)
    all_src_labels = torch.tensor(all_src_labels)

    logs = {
        **roc_info,
        "loss": total_loss,
        "loss_cls": total_loss_cls,
        "loss_mmd": total_loss_mmd,
        "loss_auc": total_loss_auc,
        "loss_soft": total_loss_soft,
        "acc_raw": acc_raw,
        "acc_per_class": acc_per_class.cpu().tolist(),
        "acc_balanced": acc_balanced,
        "probs": all_src_probs,
        "preds": all_src_preds,
        "labels": all_src_labels,
    }

    return logs


def get_auc(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    return auc_score


def get_best_threshold(labels, probs, return_arrays=True):

    # Check arguments
    if torch.sum(labels) > 0:
        return {
            "auc": 0.0,
            "best_fpr": 0.0,
            "best_tpr": 0.0,
            "best_fom": 0.0,
            "best_thr": 0.0,
        }

    # Compute ROC curve and AUC
    logger.debug("labels = %s", labels)
    logger.debug("probs = %s", probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    logger.debug("fpr = %s", fpr)
    logger.debug("tpr = %s", tpr)
    roc_auc = auc(fpr, tpr)

    # Compute Figure of Merit: FOM = TPR / sqrt(TPR + FPR)
    fom = tpr / np.sqrt(tpr + fpr + 1e-8)  # small value to avoid division by zero
    best_idx = np.argmax(fom)
    best_fpr, best_tpr, best_fom, best_thr = (
        fpr[best_idx],
        tpr[best_idx],
        fom[best_idx],
        thresholds[best_idx],
    )

    logs = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "best_fpr": best_fpr,
        "best_tpr": best_tpr,
        "best_fom": best_fom,
        "best_thr": best_thr,
    }

    if not return_arrays:
        logs.pop("fpr")
        logs.pop("tpr")

    return logs, thresholds
