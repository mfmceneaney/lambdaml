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

def train(config):
    """
    :param: config
    """
    pass

def test(config):
    """
    :param: config
    """
    pass

def apply(config):
    """
    :param: config
    """
    pass

def experiment(config,use_wanbd=True):
    """
    :param: config
    """

    # Log experiment config
    if use_wanbd: wandb.init(**config)

    # Run training validation and testing
    train_val = train(config)
    test_val  = test(config)
    apply_val = apply(config)

    # Finish experiment
    if use_wanbd: wandb.finish()

    return train_val, test_val, apply_val

def optimize(opt_par_lims,default_config):
    """
    :param: opt_config
    :param: opt_par_lims
    :param: default_config
    """

    def objective(trial):

        trial_config = default_config.copy() #NOTE: COPY IS IMPORTANT HERE!

        #TODO: Suggest trial params and substitute into trial_config

        experiment_val = experiment(trial_config)

        return experiment_val

    # Load or create pruner, sampler, and study
    pruner = optuna.pruners.MedianPruner() if opt_config['pruning'] else optuna.pruners.NopPruner()
    sampler = TPESampler() #TODO: Add command line option for selecting different sampler types.
    study = optuna.create_study(
        storage='sqlite:///'+opt_config['db_path'],
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        direction="minimize",
        load_if_exists=True
    ) #TODO: Add options for different SQL programs: Postgre, MySQL, etc.

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.ntrials,
        timeout=args.timeout,
        gc_after_trial=True
    ) #NOTE: gc_after_trial=True is to avoid OOM errors see https://optuna.readthedocs.io/en/stable/faq.html#out-of-memory-gc-collect
