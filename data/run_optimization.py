#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

import argparse
import numpy as np
import yaml
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.nn import CrossEntropyLoss

# Local imports
from data import static_split, CustomDataset
from models import GIN
from utils import experiment

def main(
        root_labelled="",
        root_unlabelled="",
        lengths_labelled=[0.8,0.1,0.1],
        lengths_unlabelled=None,
        num_workers=0,
        max_files=0,
        use_weighted_samplers=False,
        epochs=100,
        opt_par_config={},
        use_wandb=True,
        project='project',
        direction='minimize',
        minimization_key='roc_auc',
        load_if_exists=True,
        ntrials=1,
        timeout=864000,
        gc_after_trial=True,
        log_dir="./",
    ):
    """
    :description: Run an optuna optimization experiment training, testing, and applying a Lambda GNN model and suggesting new hyperparameters based on the selected optimization algorithm.

    :param: root_labelled
    :param: root_unlabelled
    :param: lengths_labelled
    :param: lengths_unlabelled
    :param: num_workers
    :param: max_files
    :param: use_weighted_samplers
    :param: epochs
    :param: opt_par_config
    :param: use_wandb
    :param: project
    :param: direction
    :param: minimization_key
    :param: load_if_exists
    :param: ntrials
    :param: timeout
    :param: gc_after_trial
    :param: log_dir
    """

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    ds_labelled = CustomDataset(
            root_labelled,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            max_files=max_files,
        )
    ds_unlabelled = CustomDataset(
            root_unlabelled,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            max_files=max_files,
        )

    # Split datasets
    ds_labelled_train, ds_labelled_val, ds_labelled_test = static_split(ds_labelled,lengths_labelled)

    # Create samplers if requested
    sl_labelled_train     = None
    sl_labelled_val       = None
    if use_weighted_samplers:
        _, train_counts = np.unique(ds_labelled_train.y, return_counts=True)
        train_weights = [1/train_counts[i] for i in train_dataset.y]
        train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(ds_labelled_train), replacement=True)
        _, val_counts = np.unique(ds_labelled_val.y, return_counts=True)
        val_weights = [1/val_counts[i] for i in ds_labelled_val.y]
        val_sampler = WeightedRandomSampler(weights=val_weights, num_samples=len(ds_labelled_val), replacement=True)

    # Create dataloaders
    dl_labelled_train   = DataLoader(ds_labelled_train, sampler=sl_labelled_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_labelled_val     = DataLoader(ds_labelled_val, sampler=sl_labelled_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_labelled_test    = DataLoader(ds_labelled_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_unlabelled_apply = DataLoader(ds_unlabelled, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create model #TODO: Could load args from yaml so easily configurable
    model_params = {
            'in_channels': 7, #ds_labelled_train.num_node_features,
            'gnn_num_layers': 4,
            'gnn_num_mlp_layers': 3,
            'gnn_mlp_hidden_dim': 128,
            'gnn_mlp_norm': 'batch_norm',
            'gnn_mlp_act': 'relu',
            'train_epsilon': False,
            'head_num_mlp_layers': 3,
            'head_mlp_hidden_dim':  128,
            'head_norm': None,
            'head_act': 'relu',
            'dropout': 0.5,
            'out_channels': 2, #ds_labelled_train.num_classes,
            'pool': 'max',
    }
    model=GIN(
        **model_params
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create loss function with weights
    data_labels    = ds_labelled.current_ds.y
    unique, counts = np.unique(data_labels,return_counts=True)
    weight_signal  = counts[1]/counts[0]
    weight         = torch.FloatTensor([weight_signal, 1.0]).to(device)
    criterion      = CrossEntropyLoss(weight=weight if not use_weighted_samplers else None,reduction='mean')

    # Set miscellaneous parameters
    scheduler  = None
    kin_names  = ["idxe","idxp","idxpi","Q2","nu","W","x","y","z_ppim","xF_ppim","mass_ppim"]
    kin_labels = ["idxe","idxp","idxpi","$Q^2$ (GeV$^2$)","$\\nu$","$W$ (GeV)","$x$","$y$","$z_{p\pi^{-}}","$x_{F p\pi^{-}}","$M_{p\pi^{-}}$ (GeV)"]

    # Create config
    config = {
        "model": model,
        "device": device,
        "train_dataloader": dl_labelled_train,
        "val_dataloader": dl_labelled_val,
        "test_dataloader": dl_labelled_test,
        "apply_dataloader": dl_unlabelled_apply,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "epochs": epochs,
        "kin_names": kin_names,
        "kin_labels": kin_labels,
    }

    wandb_config = {
        "device": device,
        "root_labelled": root_labelled,
        "root_unlabelled": root_unlabelled,
        "lengths_labelled": lengths_labelled,
        "lengths_unlabelled": lengths_unlabelled,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "num_workers": num_workers,
        "max_files": max_files,
        **model_params
    }

    # Run optimize with the optuna framework
    optimize(
        config=config,
        opt_par_config=opt_par_config,
        use_wandb=use_wandb,
        wandb_project=project,
        wandb_config=wandb_config
        study_name=project,
        direction=direction,
        minimization_key=minimization_key,
        load_if_exists=load_if_exists,
        ntrials=ntrials,
        timeout=timeout,
        gc_after_trial=gc_after_trial,
        log_dir=log_dir,
        )

# Run script
if __name__=="__main__":
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run a binary classification experiment: training, testing, and applying the model.')

    # Add arguments
    parser.add_argument('--ds_labelled', type=str, default='',
                        help='Path to labelled dataset')
    parser.add_argument('--ds_unlabelled', type=str, default='',
                        help='Path to unlabelled dataset')
    parser.add_argument('--lengths_labelled', type=float, default=[0.8,0.1,0.1], nargs=3,
                        help='Split fractions for labelled dataset')
    parser.add_argument('--lengths_unlabelled', type=float, default=None, nargs=3,
                        help='Split fractions for unlabelled dataset')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers processes for dataloaders')
    parser.add_argument('--max_files', type=int, default=0,
                        help='Maximum number of files to use from dataset')
    parser.add_argument('--use_weighted_samplers', action='store_true',
                        help='Use weighted samplers instead of loss weighting for imbalanced datasets')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for which to train')
    parser.add_argument('--opt_par_config_path', type=str, default='opt_par_config.yaml',
                        help='Path to optimization parameters yaml config file')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Log to WANDB')
    parser.add_argument('--project', type=str, default='project',
                        help='WANDB project name')
    parser.add_argument('--direction', type=str, default='minimize',
                        help='Optuna study optimization direction')
    parser.add_argument('--minimization_key', type=str, default='roc_auc',
                        help='Optuna study minimization metric key')
    parser.add_argument('--load_if_exists', action='store_true',
                        help='Optuna study load if exists')
    parser.add_argument('--ntrials', type=int, default=100,
                        help='Optuna study maximum number of trials')
    parser.add_argument('--timeout', type=int, default=864000,
                        help='Optuna study maximum time for project to complete')
     parser.add_argument('--gc_after_trial', action='store_true',
                        help='Optuna study run garbage collection after each trial')
    parser.add_argument('--log_dir', type=str, default='./',
                        help='Log directory path')

    # Parse
    args = parser.parse_args()

    # Get optimization configuration from yaml file
    opt_par_config = {}
    with open(yaml_path) as f:
        opt_par_config = yaml.safe_load(f)

    # Run
    main(
            root_labelled=args.ds_labelled,
            root_unlabelled=args.ds_unlabelled,
            lengths_labelled=args.lengths_labelled,
            lengths_unlabelled=args.lengths_unlabelled,
            num_workers=args.num_workers,
            max_files=args.max_files,
            use_weighted_samplers=args.use_weighted_samplers, #TODO: Add this to args
            epochs=args.epochs,
            opt_par_config=opt_par_config,#TODO: load from yaml or just set to empty and if empty do some defaults? ... do you want to optimize batch and learning_rate????!?!?!?  That's going to be complicated given current code structure.
            use_wandb=args.use_wandb,
            project=args.project,
            direction=args.direction,
            minimization_key=args.minimization_key,
            load_if_exists=args.load_if_exists,
            ntrials=args.ntrials,
            timeout=args.timeout,
            gc_after_trial=args.gc_after_trial,
            log_dir=args.log_dir,
        )
