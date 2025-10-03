# TRAIN
# pylint: disable=no-member
from tqdm import tqdm
import torch
import optuna

# Local imports
from .functional import (
    sigmoid_growth,
    gen_soft_labels,
    loss_da,
    loss_can,
    loss_titok,
)
from .validate import (
    val_da,
    val_can,
    val_titok,
)
from .log import setup_logger


# Set module logger
logger = setup_logger(__name__)


def train_da(
    encoder,
    clf,
    disc,
    optimizer,
    src_train_loader,
    tgt_train_loader,
    src_val_loader,
    tgt_val_loader,
    num_classes=2,
    sg_idx=1,
    nepochs=100,
    lr_scheduler=None,
    alpha_fn=0.1,
    device="cuda:0",
    verbose=True,
    trial=None,
    metric_fn=lambda logs: logs[0]["auc"],  # Available logs are [val_logs]
):

    # Set models in train mode
    encoder.train()
    clf.train()
    disc.train()

    # Set logging lists to return
    logs = {}
    logs["train_aucs"] = []
    logs["train_losses"] = []
    logs["train_losses_cls"] = []
    logs["train_losses_disc"] = []
    logs["train_accs_raw"] = []
    logs["train_accs_per_class"] = []
    logs["train_accs_balanced"] = []
    logs["val_aucs"] = []
    logs["val_losses"] = []
    logs["val_losses_cls"] = []
    logs["val_losses_disc"] = []
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
            loss, _, _ = loss_da(
                src_logits,
                src_labels,
                dom_logits,
                dom_labels,
            )

            # Backpropagate losses and update parameters
            loss.backward()
            optimizer.step()

        # Evaluate on training and vallidation data and then put model back in training mode
        train_logs = val_da(
            encoder,
            clf,
            disc,
            optimizer,
            src_train_loader,
            tgt_train_loader,
            num_classes=num_classes,
            sg_idx=sg_idx,
            alpha=alpha,
            device=device,
        )
        val_logs = val_da(
            encoder,
            clf,
            disc,
            optimizer,
            src_val_loader,
            tgt_val_loader,
            num_classes=num_classes,
            sg_idx=sg_idx,
            alpha=alpha,
            device=device,
        )
        encoder.train()
        clf.train()
        disc.train()

        # Optionally prune if using optuna
        if trial is not None:

            # Compute metric and report
            metric = metric_fn([val_logs])
            logger.debug("Reporting metric to optuna trial: %f", metric)
            trial.report(metric, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Append metrics for logging
        logs["train_aucs"].append(train_logs["auc"])
        logs["train_losses"].append(train_logs["loss"])
        logs["train_losses_cls"].append(train_logs["loss_cls"])
        logs["train_losses_disc"].append(train_logs["loss_disc"])
        logs["train_accs_raw"].append(train_logs["acc_raw"])
        logs["train_accs_per_class"].append(train_logs["acc_per_class"])
        logs["train_accs_balanced"].append(train_logs["acc_balanced"])
        logs["val_aucs"].append(val_logs["auc"])
        logs["val_losses"].append(val_logs["loss"])
        logs["val_losses_cls"].append(val_logs["loss_cls"])
        logs["val_losses_disc"].append(val_logs["loss_disc"])
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

    return logs


def train_can(
    encoder,
    clf,
    projector,
    optimizer,
    src_train_loader,
    tgt_train_loader,
    src_val_loader,
    tgt_val_loader,
    num_classes=2,
    sg_idx=1,
    nepochs=100,
    lr_scheduler=None,
    temp_fn=0.1,
    alpha_fn=0.1,
    device="cuda:0",
    verbose=True,
    trial=None,
    metric_fn=lambda logs: logs[0]["auc"],  # Available logs are [val_logs]
):

    # Set models in train mode
    encoder.train()
    clf.train()
    projector.train()

    # Set logging lists to return
    logs = {}
    logs["train_aucs"] = []
    logs["train_losses"] = []
    logs["train_losses_cls"] = []
    logs["train_losses_can"] = []
    logs["train_accs_raw"] = []
    logs["train_accs_per_class"] = []
    logs["train_accs_balanced"] = []
    logs["val_aucs"] = []
    logs["val_losses"] = []
    logs["val_losses_cls"] = []
    logs["val_losses_can"] = []
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

        # Iterate over source and target loaders in parallel
        for src_batch, tgt_batch in zip(src_train_loader, tgt_train_loader):

            # Reset gradients
            optimizer.zero_grad()

            # Source graph forward pass
            src_batch = src_batch.to(device)
            src_feats = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_logits = clf(src_feats)
            src_projs = projector(src_feats)
            src_labels = src_batch.y

            # Target graph forward pass
            tgt_batch = tgt_batch.to(device)
            tgt_feats = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)
            tgt_projs = projector(tgt_feats)

            # Compute loss
            loss, _, _ = loss_can(
                src_logits,
                src_labels,
                src_projs,
                tgt_projs,
                alpha=alpha,
                temp=temp,
            )

            # Backpropagate losses and update parameters
            loss.backward()
            optimizer.step()

        # Evaluate on training and vallidation data and then put model back in training mode
        train_logs = val_can(
            encoder,
            clf,
            projector,
            optimizer,
            src_train_loader,
            tgt_train_loader,
            num_classes=num_classes,
            sg_idx=sg_idx,
            temp=temp,
            alpha=alpha,
            device=device,
        )
        val_logs = val_can(
            encoder,
            clf,
            projector,
            optimizer,
            src_val_loader,
            tgt_val_loader,
            num_classes=num_classes,
            sg_idx=sg_idx,
            temp=temp,
            alpha=alpha,
            device=device,
        )
        encoder.train()
        clf.train()
        projector.train()

        # Optionally prune if using optuna
        if trial is not None:

            # Compute metric and report
            metric = metric_fn([val_logs])
            logger.debug("Reporting metric to optuna trial: %f", metric)
            trial.report(metric, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Append metrics for logging
        logs["train_aucs"].append(train_logs["auc"])
        logs["train_losses"].append(train_logs["loss"])
        logs["train_losses_cls"].append(train_logs["loss_cls"])
        logs["train_losses_con"].append(train_logs["loss_con"])
        logs["train_accs_raw"].append(train_logs["acc_raw"])
        logs["train_accs_per_class"].append(train_logs["acc_per_class"])
        logs["train_accs_balanced"].append(train_logs["acc_balanced"])
        logs["val_aucs"].append(val_logs["auc"])
        logs["val_losses"].append(val_logs["loss"])
        logs["val_losses_cls"].append(val_logs["loss_cls"])
        logs["val_losses_con"].append(val_logs["loss_con"])
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

    return logs


def train_titok(
    encoder,
    clf,
    optimizer,
    src_train_loader,
    tgt_train_loader,
    src_val_loader,
    tgt_val_loader,
    num_classes=2,
    sg_idx=1,
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
    trial=None,
    metric_fn=lambda logs: logs[0]["auc"],  # Available logs are [val_logs]
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
    logs["train_aucs"] = []
    logs["train_losses"] = []
    logs["train_losses_cls"] = []
    logs["train_losses_auc"] = []
    logs["train_losses_mmd"] = []
    logs["train_losses_soft"] = []
    logs["train_accs_raw"] = []
    logs["train_accs_per_class"] = []
    logs["train_accs_balanced"] = []
    logs["val_aucs"] = []
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
            sg_idx=sg_idx,
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
            sg_idx=sg_idx,
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

        # Optionally prune if using optuna
        if trial is not None:

            # Compute metric and report
            metric = metric_fn([val_logs])
            logger.debug("Reporting metric to optuna trial: %f", metric)
            trial.report(metric, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Append metrics for logging
        logs["train_aucs"].append(train_logs["auc"])
        logs["train_losses"].append(train_logs["loss"])
        logs["train_losses_cls"].append(train_logs["loss_cls"])
        logs["train_losses_mmd"].append(train_logs["loss_mmd"])
        logs["train_losses_auc"].append(train_logs["loss_auc"])
        logs["train_losses_soft"].append(train_logs["loss_soft"])
        logs["train_accs_raw"].append(train_logs["acc_raw"])
        logs["train_accs_per_class"].append(train_logs["acc_per_class"])
        logs["train_accs_balanced"].append(train_logs["acc_balanced"])
        logs["val_aucs"].append(val_logs["auc"])
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
