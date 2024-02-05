#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

import argparse
import hipopy.hipopy as hp
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch.nn import CrossEntropyLoss
import tqdm

# Local imports
from data import CustomDataset
from models import GIN
from utils import experiment

def main(root_labelled="",root_unlabelled="",lengths_labelled=[0.8,0.1,0.1],lengths_unlabelled=None,batch_size=32,lr=1e-3,epochs=100,use_wandb=True,num_workers=0):
    """
    :description: Create PyG dataset and save to file.  Graph data is taken from REC::Traj and preprocessed with processing.preprocess_rec_traj.

    :param: root_labelled
    :param: root_unlabelled
    :param: lengths_labelled
    :param: lengths_unlabelled
    :param: batch_size
    :param: lr
    :param: epochs
    :param: use_wandb
    :param: num_workers
    """

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    ds_labelled = CustomDataset(
            root_labelled,
            transform=None,
            pre_transform=None,
            pre_filter=None
        )
    ds_unlabelled = CustomDataset(
            root_unlabelled,
            transform=None,
            pre_transform=None,
            pre_filter=None
        )

    # Split datasets
    ds_labelled_train, ds_labelled_val, ds_labelled_test = random_split(ds_labelled,lengths_labelled)

    # Create samplers if requested #TODO!
    use_weighted_samplers = False
    sl_labelled_train     = None
    sl_labelled_val       = None

    # Create dataloaders
    dl_labelled_train   = DataLoader(ds_labelled_train, sampler=sl_labelled_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_labelled_val     = DataLoader(ds_labelled_val, sampler=sl_labelled_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_labelled_test    = DataLoader(ds_labelled_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_unlabelled_apply = DataLoader(ds_unlabelled, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create model #TODO: Could load args from yaml so easily configurable
    model_params = {
            'in_channels': 7,
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
            'out_channels': 2,
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
    kin_names  = ["Q2","W","x","y","z_ppim","xF_ppim","mass_ppim"]
    kin_labels = ["$Q^2$ (GeV$^2$)","$W$ (GeV)","$x$","$y$","$z_{p\pi^{-}}","$x_{F p\pi^{-}}","$M_{p\pi^{-}}$ (GeV)"]

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

    # Run experiment
    experiment(config,use_wandb=use_wandb)

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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for dataloaders')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for which to train')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Log to WANDB')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers processes for dataloaders')

    # Parse
    args = parser.parse_args()

    # Run
    main(
            root_labelled=args.ds_labelled,
            root_unlabelled=args.ds_unlabelled,
            lengths_labelled=args.lengths_labelled,
            lengths_unlabelled=args.lengths_unlabelled,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            epochs=args.epochs,
            use_wandb=args.use_wandb,
            num_workers=args.num_workers,
        )
