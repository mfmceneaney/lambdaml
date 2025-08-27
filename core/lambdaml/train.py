# TRAIN
# pylint: disable=no-member
from tqdm import tqdm

# Local imports
from .functional import sigmoid_growth, gen_soft_labels, loss_titok
from .validate import val_titok


# def train(epochs=100, alpha_fn=0.1):
#     encoder.train()
#     clf.train()
#     disc.train()

#     # Set logging lists to return
#     clf_losses = []
#     dom_losses = []
#     clf_accs = []
#     lrs = []

#     # Loop training epochs
#     for epoch in range(1, epochs + 1):

#         # Check alpha function
#         if callable(alpha_fn):
#             alpha = alpha_fn(epoch, epochs)
#         else:
#             alpha = alpha_fn

#         total_clf_loss = 0
#         total_domain_loss = 0
#         # Parallel iteration over source and target loaders
#         for src_batch, tgt_batch in zip(src_loader, tgt_loader):
#             optimizer.zero_grad()

#             # Source graph forward pass
#             src_batch = src_batch.to(device)
#             src_emb = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
#             src_out = clf(src_emb)
#             src_loss = F.cross_entropy(src_out, src_batch.y)

#             # Target graph forward pass
#             tgt_batch = tgt_batch.to(device)
#             tgt_emb = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)

#             # Domain classification loss (labels: 0 for source, 1 for target)
#             domain_emb = torch.cat([src_emb, tgt_emb], dim=0)
#             domain_labels = torch.cat(
#                 [
#                     torch.zeros(src_emb.size(0), dtype=torch.long),
#                     torch.ones(tgt_emb.size(0), dtype=torch.long),
#                 ],
#                 dim=0,
#             ).to(device)

#             domain_pred = disc(domain_emb, alpha)
#             domain_loss = F.cross_entropy(domain_pred, domain_labels)

#             loss = src_loss + domain_loss
#             loss.backward()
#             optimizer.step()

#             total_clf_loss += src_loss.item()
#             total_domain_loss += domain_loss.item()

#         # Get accuracy
#         src_acc, _, _ = eval_model(src_loader)
#         encoder.train()
#         clf.train()

#         # Append metrics for logging
#         clf_losses.append(total_clf_loss)
#         dom_losses.append(total_domain_loss)
#         clf_accs.append(src_acc)

#         # Log and step learning rate scheduler
#         lrs.append(optimizer.param_groups[0]["lr"])
#         scheduler.step()

#         print(
#             f"Epoch {epoch:03d}  Classifier Loss: {total_clf_loss:.4f}  Discriminator Loss: {total_domain_loss:.4f}"
#         )

#     return clf_losses, dom_losses, clf_accs, lrs


# def train_can(epochs=100, temp_fn=temp_fn, alpha_fn=0.1):
#     encoder.train()
#     clf.train()

#     # Set logging lists to return
#     clf_losses = []
#     can_losses = []
#     clf_accs = []
#     lrs = []

#     # Loop training epochs
#     for epoch in range(1, epochs + 1):

#         # Check alpha function
#         if callable(alpha_fn):
#             alpha = alpha_fn(epoch, epochs)
#         else:
#             alpha = alpha_fn

#         # Check temp function
#         if callable(temp_fn):
#             temp = temp_fn(epoch, epochs)
#         else:
#             temp = temp_fn

#         total_clf_loss = 0
#         total_can_loss = 0
#         # Parallel iteration over source and target loaders
#         for src_batch, tgt_batch in zip(src_loader, tgt_loader):
#             optimizer.zero_grad()

#             # Source graph forward pass
#             src_batch = src_batch.to(device)
#             src_emb = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
#             src_out = clf(src_emb)
#             src_loss = F.cross_entropy(src_out, src_batch.y)

#             # Target graph forward pass
#             tgt_batch = tgt_batch.to(device)
#             tgt_emb = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)

#             # Contrastive loss (align source and target representations)
#             z1 = projector(src_emb)
#             z2 = projector(tgt_emb)
#             can_loss = contrastive_loss(z1, z2, temperature=temp)

#             # # Classification loss (only on source)
#             # cls_loss = F.cross_entropy(src_out, src_batch.y)

#             loss = src_loss + alpha * can_loss
#             loss.backward()
#             optimizer.step()

#             total_clf_loss += src_loss.item()
#             total_can_loss += can_loss.item()

#         # Get accuracy
#         src_acc, _, _ = eval_model(src_loader_unweighted)
#         encoder.train()
#         clf.train()

#         # Append metrics for logging
#         clf_losses.append(total_clf_loss)
#         can_losses.append(total_can_loss)
#         clf_accs.append(src_acc)

#         # Log and step learning rate scheduler
#         lrs.append(optimizer.param_groups[0]["lr"])
#         if scheduler is not None:
#             scheduler.step()

#         print(
#             f"Epoch {epoch:03d}  Classifier Loss: {total_clf_loss:.4f}  Contrastive Loss: {total_can_loss:.4f}"
#         )

#     return clf_losses, can_losses, clf_accs, lrs


def train_titok(
    encoder,
    clf,
    optimizer,
    src_train_loader,
    tgt_train_loader,
    src_val_loader,
    tgt_val_loader,
    num_classes=2,
    soft_labels_temp=2,
    nepochs=100,
    lr_scheduler=None,
    confidence_threshold=0.8,
    temp_fn=0.1,
    alpha_fn=0.1,
    lambda_fn=sigmoid_growth,
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
            loss, _, _, _, _ = loss_titok(
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

            # Backpropagate losses and update parameters
            loss.backward()
            optimizer.step()

        # Evaluate on training and vallidation data and then put model back in training mode
        train_logs = val_titok(
            encoder,
            clf,
            optimizer,
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
        )
        val_logs = val_titok(
            encoder,
            clf,
            optimizer,
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
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Print training info
        if verbose:
            message = [f"Epoch {epoch:03d}"]
            for key in logs:
                if type(logs[key][-1]) == float:
                    message.append(f"{key}: {logs[key][-1]:.4f}")
            message = "\n\t".join(message)
            print(message)

    return logs, soft_labels
