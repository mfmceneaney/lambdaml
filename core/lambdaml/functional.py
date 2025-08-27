# TRAIN
# pylint: disable=no-member
import numpy as np
import torch
import torch.nn.functional as F


def sigmoid_growth(epoch, nepochs, coefficient=0.05):
    return coefficient * (2.0 / (1.0 + np.exp(-10 * epoch / nepochs)) - 1)


def sigmoid_decay(epoch, nepochs, coefficient=0.05):
    return sigmoid_growth(nepochs - epoch - 1, nepochs, coefficient)


def linear_growth(epoch, nepochs, t_min=0.07, t_max=0.5):
    return t_min + (t_max - t_min) * (epoch / nepochs)


def linear_decay(epoch, nepochs, t_min=0.07, t_max=0.5):
    return t_min + (t_max - t_min) * (1 - epoch / nepochs)


def contrastive_loss(z1, z2, temperature=0.5):

    # Compute the contrastive loss: NT-Xent (simplified)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)

    representations = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(representations, representations.T)

    sim_ij = torch.diag(similarity, batch_size)
    sim_ji = torch.diag(similarity, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nominator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(similarity / temperature), dim=1) - torch.exp(
        torch.ones_like(positives) / temperature
    )
    loss = -torch.log(nominator / denominator)

    return loss.mean()


# ----- TIToK Loss definitions -----#


def exploss(y_source_prob, y_source, alpha=0.5):

    # Compute the exponential loss
    loss_sum = 0
    nc = y_source_prob.size(1)
    for i in range(nc):
        index_i = y_source == i
        a = torch.exp(-alpha * y_source_prob[index_i, i])
        b = 0
        for j in range(nc):
            if j == i:
                continue
            index_j = y_source == j
            ni = index_i.float().sum().item()
            nj = index_j.float().sum().item()
            if ni > 0 and nj > 0:
                b += torch.sum(torch.exp(alpha * y_source_prob[index_j, i])) / (ni * nj)
        loss_sum += torch.sum(a) * b

    return loss_sum


def soft_label_loss(tgt_logits, soft_labels_batch, temperature=2.0):

    # Compute the soft label loss
    loss_soft = torch.zeros(())
    output = F.softmax(tgt_logits / temperature, dim=1)
    if float(output.size(0)) > 0:
        loss_soft = -torch.sum(soft_labels_batch * torch.log(output)) / output.size(0)

    return loss_soft


def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5):
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), -1, -1)
    total1 = total.unsqueeze(1).expand(-1, total.size(0), -1)
    L2_distance = ((total0 - total1) ** 2).sum(2)

    bandwidth = torch.sum(L2_distance.data) / (total.size(0) ** 2 - total.size(0))
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]

    return sum(kernel_val)


def mmd_loss(source, target):

    # Compute the Maximum Mean Discrepancy loss
    batch_size = source.size(0)
    kernels = gaussian_kernel(source, target)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)

    return loss


def gen_soft_labels(
    num_classes, loader, encoder, clf, temperature=2, device="cuda:0"
):  # NOTE: FROM: https://github.com/big-chuan-bro/TiTok/blob/main/ImbalancedDA/Titok.py

    # Set models to evaluation mode
    encoder.eval()
    clf.eval()

    # Create arrays
    soft_labels = torch.zeros(num_classes, 1, num_classes).to(device)
    sum_classes = torch.zeros(num_classes).to(device)
    preds_total = []
    label_total = []

    # Loop data
    for batch in loader:

        # Apply model
        batch = batch.to(device)
        feats = encoder(batch.x, batch.edge_index, batch.batch)
        logits = clf(feats)

        # Add to arrays
        label_total.append(batch.y)
        preds = F.softmax(logits / temperature, dim=1).data.to(device)
        preds_total.append(preds)

    # Concatenate arrays
    preds_total = torch.cat(preds_total)
    label_total = torch.cat(label_total)

    # Loop data and set class counts and soft labels
    for i in range(len(loader)):
        sum_classes[label_total[i]] += 1
        soft_labels[label_total[i]][0] += preds_total[i]

    # Loop classes and divide soft labels by class counts
    for cl_idx in range(num_classes):
        soft_labels[cl_idx][0] /= sum_classes[cl_idx]

    return soft_labels


def ret_soft_label(
    label, soft_labels, num_classes=2, device="cuda:0"
):  # NOTE: FROM: https://github.com/big-chuan-bro/TiTok/blob/main/ImbalancedDA/Titok.py

    # Compute the soft label for a batch
    soft_label_for_batch = torch.zeros(label.size(0), num_classes).to(device)
    for i in range(label.size(0)):
        soft_label_for_batch[i] = soft_labels[label.data[i]]

    return soft_label_for_batch


def loss_titok(
    src_feats,
    src_logits,
    src_labels,
    tgt_feats,
    tgt_logits,
    soft_labels,
    loss_auc_alpha=0.5,
    loss_soft_temperature=2.0,
    confidence_threshold=0.8,
    num_classes=2,
    pretraining=False,
    device="cuda:0",
    coeff_mmd=0.3,
    lambd=1.0,
    coeff_auc=0.01,
    coeff_soft=0.25,
):

    # Source classification loss
    loss_cls = F.cross_entropy(src_logits, src_labels)

    # Check if pretraining for soft labels
    if pretraining:
        return loss_cls, loss_cls, torch.zeros(()), torch.zeros(()), torch.zeros(())

    # Apply softmax to get probabilities on target domain
    tgt_probs = F.softmax(tgt_logits, dim=1)

    # Get max class probabilities
    confidences, pred_classes = torch.max(tgt_probs, dim=1)  # [B]

    # Select samples with confidence above a threshold
    mask = confidences >= confidence_threshold

    # Select the logits and predicted labels of those confident samples
    tgt_logits_confident = tgt_logits[mask]  # [B_confident, num_classes]
    tgt_labels_confident = pred_classes[mask]  # [B_confident]

    # Get soft labels
    soft_labels_batch = ret_soft_label(
        tgt_labels_confident, soft_labels, num_classes=num_classes, device=device
    )

    # AUC-style loss (exploss)
    loss_auc = exploss(F.softmax(src_logits, dim=1), src_labels, alpha=loss_auc_alpha)

    # Optional: MMD loss between source/target embeddings
    loss_mmd = mmd_loss(src_feats, tgt_feats)

    # Target knowledge distillation loss (on confident samples only)
    loss_soft = soft_label_loss(
        tgt_logits_confident, soft_labels_batch, temperature=loss_soft_temperature
    )

    # Combine losses
    loss = (
        loss_cls
        + coeff_mmd * lambd * loss_mmd
        + coeff_auc * loss_auc
        + coeff_soft * loss_soft
    )

    return loss, loss_cls, loss_mmd, loss_auc, loss_soft
