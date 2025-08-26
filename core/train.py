# ----------------------------------------------------------------------------------------------------#
# TRAIN
from tqdm import tqdm
import torch
import torch.nn.functional as F

# Local imports
from validate import *


def alpha_fn(epoch, total_epochs, coefficient=0.05):
    return coefficient * (2.0 / (1.0 + np.exp(-10 * epoch / total_epochs)) - 1)


def temp_fn(epoch, max_epoch, t_min=0.07, t_max=0.5):
    return t_min + (t_max - t_min) * (1 - epoch / max_epoch)


def lambda_fn(epoch, epochs):
    return 2 / (1 + np.exp(-10 * epoch / epochs)) - 1


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


def train(epochs=100, alpha_fn=0.1):
    encoder.train()
    clf.train()
    disc.train()

    # Set logging lists to return
    clf_losses = []
    dom_losses = []
    clf_accs = []
    lrs = []

    # Loop training epochs
    for epoch in range(1, epochs + 1):

        # Check alpha function
        if callable(alpha_fn):
            alpha = alpha_fn(epoch, epochs)
        else:
            alpha = alpha_fn

        total_clf_loss = 0
        total_domain_loss = 0
        # Parallel iteration over source and target loaders
        for src_batch, tgt_batch in zip(src_loader, tgt_loader):
            optimizer.zero_grad()

            # Source graph forward pass
            src_batch = src_batch.to(device)
            src_emb = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_out = clf(src_emb)
            src_loss = F.cross_entropy(src_out, src_batch.y)

            # Target graph forward pass
            tgt_batch = tgt_batch.to(device)
            tgt_emb = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)

            # Domain classification loss (labels: 0 for source, 1 for target)
            domain_emb = torch.cat([src_emb, tgt_emb], dim=0)
            domain_labels = torch.cat(
                [
                    torch.zeros(src_emb.size(0), dtype=torch.long),
                    torch.ones(tgt_emb.size(0), dtype=torch.long),
                ],
                dim=0,
            ).to(device)

            domain_pred = disc(domain_emb, alpha)
            domain_loss = F.cross_entropy(domain_pred, domain_labels)

            loss = src_loss + domain_loss
            loss.backward()
            optimizer.step()

            total_clf_loss += src_loss.item()
            total_domain_loss += domain_loss.item()

        # Get accuracy
        src_acc, _, _ = eval_model(src_loader)
        encoder.train()
        clf.train()

        # Append metrics for logging
        clf_losses.append(total_clf_loss)
        dom_losses.append(total_domain_loss)
        clf_accs.append(src_acc)

        # Log and step learning rate scheduler
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        print(
            f"Epoch {epoch:03d}  Classifier Loss: {total_clf_loss:.4f}  Discriminator Loss: {total_domain_loss:.4f}"
        )

    return clf_losses, dom_losses, clf_accs, lrs


def train_can(epochs=100, temp_fn=temp_fn, alpha_fn=0.1):
    encoder.train()
    clf.train()

    # Set logging lists to return
    clf_losses = []
    can_losses = []
    clf_accs = []
    lrs = []

    # Loop training epochs
    for epoch in range(1, epochs + 1):

        # Check alpha function
        if callable(alpha_fn):
            alpha = alpha_fn(epoch, epochs)
        else:
            alpha = alpha_fn

        # Check temp function
        if callable(temp_fn):
            temp = temp_fn(epoch, epochs)
        else:
            temp = temp_fn

        total_clf_loss = 0
        total_can_loss = 0
        # Parallel iteration over source and target loaders
        for src_batch, tgt_batch in zip(src_loader, tgt_loader):
            optimizer.zero_grad()

            # Source graph forward pass
            src_batch = src_batch.to(device)
            src_emb = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_out = clf(src_emb)
            src_loss = F.cross_entropy(src_out, src_batch.y)

            # Target graph forward pass
            tgt_batch = tgt_batch.to(device)
            tgt_emb = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)

            # Contrastive loss (align source and target representations)
            z1 = projector(src_emb)
            z2 = projector(tgt_emb)
            can_loss = contrastive_loss(z1, z2, temperature=temp)

            # # Classification loss (only on source)
            # cls_loss = F.cross_entropy(src_out, src_batch.y)

            loss = src_loss + alpha * can_loss
            loss.backward()
            optimizer.step()

            total_clf_loss += src_loss.item()
            total_can_loss += can_loss.item()

        # Get accuracy
        src_acc, _, _ = eval_model(src_loader_unweighted)
        encoder.train()
        clf.train()

        # Append metrics for logging
        clf_losses.append(total_clf_loss)
        can_losses.append(total_can_loss)
        clf_accs.append(src_acc)

        # Log and step learning rate scheduler
        lrs.append(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:03d}  Classifier Loss: {total_clf_loss:.4f}  Contrastive Loss: {total_can_loss:.4f}"
        )

    return clf_losses, can_losses, clf_accs, lrs


def train_titok(
    encoder,
    clf,
    src_train_loader,
    tgt_train_loader,
    src_val_loader,
    tgt_val_loader,
    num_classes=2,
    soft_labels_temp=2,
    nepochs=100,
    confidence_threshold=0.8,
    temp_fn=0.1,
    alpha_fn=0.1,
    lambda_fn=lambda_fn,
    coeff_mmd=0.3,
    coeff_auc=0.01,
    coeff_soft=0.25,
    pretrain_frac=0.2,
    device="cuda:0",
    verbose=True,
):

    # Create soft labels #NOTE: Pretrain first
    soft_labels = None
    if pretrain_frac <= 0.0:
        soft_labels = gen_soft_labels(
            num_classes,
            src_train_loader,
            encoder,
            clf,
            temperature=soft_labels_temp,
            device=device,
        )

    # Set models in train mode
    encoder.train()
    clf.train()

    # Set logging lists to return
    logs = {}
    logs["train_losses"] = []
    logs["train_losses_cls"] = []
    logs["train_losses_auc"] = []
    logs["train_losses_mmd"] = []
    logs["train_losses_soft"] = []
    logs["train_accs_raw"] = []
    logs["train_accs_per_class"] = []
    logs["train_accs_balanced"] = []
    logs["val_losses"] = []
    logs["val_losses_cls"] = []
    logs["val_losses_auc"] = []
    logs["val_losses_mmd"] = []
    logs["val_losses_soft"] = []
    logs["val_accs_raw"] = []
    logs["val_accs_per_class"] = []
    logs["val_accs_balanced"] = []
    logs["lrs"] = []

    # Loop training epochs
    for epoch in tqdm(range(1, nepochs + 1)):

        # Check alpha function
        if callable(alpha_fn):
            alpha = alpha_fn(epoch, nepochs)
        else:
            alpha = alpha_fn

        # Check temp function
        if callable(temp_fn):
            temp = temp_fn(epoch, nepochs)
        else:
            temp = temp_fn

        # Check lambda function
        if callable(lambda_fn):
            lambd = lambda_fn(epoch, nepochs)
        else:
            lambd = lambda_fn

        # Set soft labels after pretraining
        pretraining = epoch / nepochs <= pretrain_frac and pretrain_frac > 0.0
        if soft_labels is None and not pretraining:
            soft_labels = gen_soft_labels(
                num_classes,
                src_train_loader,
                encoder,
                clf,
                temperature=soft_labels_temp,
                device=device,
            )

        # Iterate over source and target loaders in parallel
        for src_batch, tgt_batch in zip(src_train_loader, tgt_train_loader):

            # Reset gradients
            optimizer.zero_grad()

            # Source graph forward pass
            src_batch = src_batch.to(device)
            src_feats = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_logits = clf(src_feats)
            src_labels = src_batch.y

            # Target graph forward pass
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
                loss_auc_alpha=0.5,
                loss_soft_temperature=2.0,
                confidence_threshold=confidence_threshold,
                pretraining=pretraining,
                num_classes=num_classes,
                device=device,
            )

            # Backpropagate losses and update parameters
            loss.backward()
            optimizer.step()

        # Evaluate on training and vallidation data and then put model back in training mode
        train_logs = val_titok(
            encoder,
            clf,
            src_train_loader,
            tgt_train_loader,
            soft_labels,
            pretraining=pretraining,
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
            temp=temp,
            alpha=alpha,
            lambd=lambd,
            coeff_mmd=coeff_mmd,
            coeff_auc=coeff_auc,
            coeff_soft=coeff_soft,
            device=device,
            verbose=verbose,
        )
        val_logs = val_titok(
            encoder,
            clf,
            src_val_loader,
            tgt_val_loader,
            soft_labels,
            pretraining=pretraining,
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
            temp=temp,
            alpha=alpha,
            lambd=lambd,
            coeff_mmd=coeff_mmd,
            coeff_auc=coeff_auc,
            coeff_soft=coeff_soft,
            device=device,
            verbose=verbose,
        )
        encoder.train()
        clf.train()

        # Append metrics for logging
        logs["train_losses"].append(train_logs["loss"])
        logs["train_losses_cls"].append(train_logs["loss_cls"])
        logs["train_losses_mmd"].append(train_logs["loss_mmd"])
        logs["train_losses_auc"].append(train_logs["loss_auc"])
        logs["train_losses_soft"].append(train_logs["loss_soft"])
        logs["train_accs_raw"].append(train_logs["acc_raw"])
        logs["train_accs_per_class"].append(train_logs["acc_per_class"])
        logs["train_accs_balanced"].append(train_logs["acc_balanced"])
        logs["val_losses"].append(val_logs["loss"])
        logs["val_losses_cls"].append(val_logs["loss_cls"])
        logs["val_losses_mmd"].append(val_logs["loss_mmd"])
        logs["val_losses_auc"].append(val_logs["loss_auc"])
        logs["val_losses_soft"].append(val_logs["loss_soft"])
        logs["val_accs_raw"].append(val_logs["acc_raw"])
        logs["val_accs_per_class"].append(val_logs["acc_per_class"])
        logs["val_accs_balanced"].append(val_logs["acc_balanced"])
        logs["lrs"].append(optimizer.param_groups[0]["lr"])

        # Step learning rate step scheduler
        if scheduler is not None:
            scheduler.step()

        # Print training info
        if verbose:
            message = [f"Epoch {epoch:03d}"]
            for key in logs:
                if type(logs[key][-1]) == float:
                    message.append(f"{key}: {logs[key][-1]:.4f}")
            message = "\n\t".join(message)
            print(message)

    return logs, soft_labels
