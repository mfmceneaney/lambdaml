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
from tqdm import tqdm

# Local
import data
import models
import classification

def train(model=None,device=None,train_dataloader=None,val_dataloader=None,optimizer=None,criterion=None,scheduler=None,epochs=100,use_wandb=True):
    """
    :param: model
    :param: train_dataloader
    :param: val_dataloader
    :param: optimizer
    :param: criterion
    :param: scheduler
    :param: epochs
    :param: use_wandb
    """

    for epoch in tqdm(range(epochs)):

        # Train model
        classification.train(
            model,
            train_dataloader,
            optimizer,
            criterion
        )

        # Validate model
        loss, outs, preds, ys, kins = classification.test(
            model,
            val_dataloader,
            criterion,
            get_kins = False #NOTE: This just does not set kins, even though a value will still be returned.  #TODO: Think about whether this is actually wise.
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
                **optimizer.param_groups[0]['lr'], #TODO: Check this...
            })

        # Step learning rate
        if scheduler is not None: scheduler.step()

def test(model=None,device=None,dataloader=None,criterion=None,kin_names=None,kin_labels=None,use_wandb=True):
    """
    :param: model
    :param: dataloader
    :param: criterion
    :param: kin_names
    :param: kin_labels
    :param: use_wandb
    """
    loss, outs, preds, ys, kins = classification.test(
        model,
        dataloader,
        criterion,
        return_kins=True
    )

    # Get binary classification metrics
    accuracy, precision, recall, precision_n, recall_n, roc_auc, plots = classification.get_binary_classification_metrics(
                                                                                outs,
                                                                                preds,
                                                                                ys,
                                                                                kins=kins,
                                                                                get_plots=True,
                                                                                kin_names=kin_names,
                                                                                kin_labels=kin_labels
                                                                            )
    mass_fit_metrics = classification.get_lambda_mass_fit(preds,true_labels=ys)#TODO: CHECK THAT ACTUALLY WANT THIS HERE

    # Log to wandb
    if use_wandb:
        wandb.log({
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'roc_auc':roc_auc,
            **plots,
            **massfit_metrics,
        })

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        **massfit_metrics
    }


def apply(model=None,device=None,dataloader=None,kin_names=None,kin_labels=None,use_wandb=True):
    """
    :param: model
    :param: dataloader
    :param: kin_names
    :param: kin_labels
    :param: use_wandb
    """
    loss, outs, preds, kins = classification.test_nolabels(
        model,
        dataloader,
    )

    # Get binary classification metrics
    metrics, plots = classification.get_binary_classification_metrics_nolabels(outs,preds,kins=kins,get_plots=True,kin_names=kin_names,kin_labels=kin_labels)
    mass_fit_metrics = classification.get_lambda_mass_fit(preds,true_labels=None)#TODO: CHECK THAT ACTUALLY WANT THIS HERE

    # Log to wandb
    if use_wandb:
        wandb.log({
            **metrics,
            **plots,
            **massfit_metrics,
        })

    return massfit_metrics

def experiment(config,use_wandb=True):
    """
    :param: config
    :param: use_wandb
    """

    # Unpack config
    model            = config['model']
    device           = config['device']
    train_dataloader = config['train_dataloader']
    val_dataloader   = config['val_dataloader']
    test_dataloader  = config['test_dataloader']
    apply_dataloader = config['apply_dataloader']
    optimizer        = config['optimizer']
    criterion        = config['criterion']
    scheduler        = config['scheduler']
    epochs           = config['epochs']
    kin_names        = config['kin_names']
    kin_labels       = config['kin_labels']

    # Log experiment config
    if use_wandb: 
        wandb.init(**config)
        wandb.watch(model)

    # Run training validation and testing
    train_val = train(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs,
        use_wandb=use_wandb
    )
    test_val  = test(
        model=model,
        device=device,
        dataloader=test_dataloader,
        criterion=criterion,
        kin_names=kin_names,
        kin_labels=kin_labels,
        use_wandb=use_wandb
    )
    apply_val = apply(
        model=model,
        device=device,
        dataloader=apply_dataloader,
        kin_names=kin_names,
        kin_labels=kin_labels,
        use_wandb=use_wandb
    )

    # Finish experiment
    if use_wandb: wandb.finish()

    return train_val, test_val, apply_val

def optimize(opt_par_lims,default_config,study_name='study',direction='minimize',):
    """
    :param: opt_config
    :param: opt_par_lims
    :param: default_config
    """

    def objective(trial):

        trial_config = default_config.copy() #NOTE: COPY IS IMPORTANT HERE!

        #TODO: Suggest trial params and substitute into trial_config also set log dir name with trial param of objective

        experiment_val = experiment(trial_config)
        roc_auc = experiment_val[1]['roc_auc']#TODO: Get roc_auc from experiment values

        return 1.0-roc_auc

    # Load or create pruner, sampler, and study
    pruner = optuna.pruners.MedianPruner() if opt_config['pruning'] else optuna.pruners.NopPruner() #TODO: Add CLI options for other pruners
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
