#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

# Logging
import wandb

# Optimization
import optuna

# Data
import numpy
import torch

# ML
import torch_geometric as tg
import sklearn
import modeloss
import umap

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Miscellaneous
import os
import sys
import tqdm

# Local
import data
import models
import classification

def train(model,dataloader,optimizer,criterion,epochs=100,scheduler=None,log_path="",use_wandb=True):
    """
    :param: config
    """

    logs = []
    for epoch in tqdm(range(epochs)):

        # Train model
        classification.train(
            model,
            dataloader,
            optimizer,
            criterion
        )

        # Validate model
        loss, outs, preds, ys, kins = classification.test(
            model,
            dataloader,
            criterion,
            get_kins = False
        )

        # Get binary classification metrics
        accuracy, precision, recall, precision_n, recall_n, roc_auc, plots = classification.get_binary_classification_metrics(preds,ys,kins=None,get_plots=False)

        # Log to wandb
        if use_wandb:
            wandb.log({
                'accuracy':accuracy,
                'precision':precision,
                'recall':recall,
                'roc_auc':roc_auc,
                **optimizer.param_groups[0]['lr'], #NOTE: Check this...
            })

        # Step learning rate
        if scheduler is not None: scheduler.step()


def test(model,dataloader,criterion,use_wandb=True):
    """
    :param: config
    """
    loss, outs, preds, ys, kins = classification.test(
        model,
        dataloader,
        criterion,
        return_kins=True
    )

    # Get binaray classification metrics
    accuracy, precision, recall, precision_n, recall_n, roc_auc, plots = classification.get_binary_classification_metrics(preds,ys,return_plots=True)

    # Log to wandb
    if use_wandb:
        wandb.log({
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'roc_auc':roc_auc,
            **plots
        })


def apply(config,use_wandb=True):
    """
    :param: config
    """
    loss, outs, preds, kins = classification.test_nolabels(
        model,
        dataloader,
    )

def experiment(config,use_wanbd=True):
    """
    :param: config
    """

    # Unpack config
    model = config['model']

    # Log experiment config
    if use_wanbd:
        wandb.init(**config)
        wandb.watch(model)

    # Run training validation and testing
    train_val = train(config,use_wandb=use_wandb)
    test_val  = test(config,use_wandb=use_wandb)
    apply_val = apply(config,use_wandb=use_wandb)

    # Finish experiment
    if use_wanbd: wandb.finish()

    return train_val, test_val, apply_val

def optimize(opt_par_lims,default_config,study_name="study",direction="minimize",):
    """
    :param: opt_config
    :param: opt_par_lims
    :param: default_config
    """

    def objective(trial):

        trial_config = default_config.copy() #NOTE: COPY IS IMPORTANT HERE!

        #TODO: Suggest trial params and substitute into trial_config

        experiment_val = experiment(trial_config)
        roc_auc = 

        return 1.0-roc_auc

    # Load or create pruner, sampler, and study
    pruner = optuna.pruners.MedianPruner() if opt_config['pruning'] else optuna.pruners.NopPruner()
    sampler = TPESampler() #TODO: Add command line option for selecting different sampler types.
    study = optuna.create_study(
        storage='sqlite:///'+opt_config['db_path'],
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        direction=direction,
        load_if_exists=True
    ) #TODO: Add options for different SQL programs: Postgre, MySQL, etc.

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True
    ) #NOTE: gc_after_trial=True is to avoid OOM errors see https://optuna.readthedocs.io/en/stable/faq.html#out-of-memory-gc-collect
