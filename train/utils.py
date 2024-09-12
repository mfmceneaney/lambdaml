#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

# ML
import torch

# Logging
import wandb

# Miscellaneous
import os.path as osp
from tqdm import tqdm

# Local
import data
import classification

def train(model=None,device=None,train_dataloader=None,val_dataloader=None,optimizer=None,criterion=None,scheduler=None,epochs=100,use_wandb=True):
    """
    :param: model
    :param: device
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
            device,
            train_dataloader,
            optimizer,
            criterion
        )

        # Validate model
        loss, outs, preds, ys, kins = classification.test(
            model,
            device,
            val_dataloader,
            criterion,
            get_kins = False #NOTE: This just does not set kins, even though a value will still be returned.  #TODO: Think about whether this is actually wise.
        )

        # Get binary classification metrics
        accuracy, precision, recall, precision_n, recall_n, roc_auc, roc_c, plots = classification.get_binary_classification_metrics(outs,preds,ys,kins=None,get_plots=False)

        # Log to wandb
        if use_wandb:
            wandb.log({
                'accuracy':accuracy,
                'precision':precision,
                'recall':recall,
                'roc_auc':roc_auc,
                'lr': optimizer.param_groups[0]['lr'], #TODO: Check this...
            })

        # Step learning rate
        if scheduler is not None: scheduler.step()

def test(model=None,device=None,dataloader=None,criterion=None,kin_names=None,kin_labels=None,use_wandb=True):
    """
    :param: model
    :param: device
    :param: dataloader
    :param: criterion
    :param: kin_names
    :param: kin_labels
    :param: use_wandb

    :return: metrics
    """
    loss, outs, preds, ys, kins = classification.test(
        model,
        device,
        dataloader,
        criterion,
        get_kins=True
    )

    # Get binary classification metrics
    accuracy, precision, recall, precision_n, recall_n, roc_auc, roc_c, plots = classification.get_binary_classification_metrics(
                                                                                outs,
                                                                                preds,
                                                                                ys,
                                                                                kins=kins,
                                                                                get_plots=True,
                                                                                kin_names=kin_names,
                                                                                kin_labels=kin_labels
                                                                            )
    massfit_metrics = classification.get_lambda_mass_fit(kins,true_labels=ys,mass_index=-1)#TODO: CHECK THAT ACTUALLY WANT THIS HERE

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
    :param: device
    :param: dataloader
    :param: kin_names
    :param: kin_labels
    :param: use_wandb

    :return: massfit_metrics
    """
    outs, preds, kins = classification.test_nolabels(
        model,
        device,
        dataloader,
    )

    # Get binary classification metrics
    metrics, plots = classification.get_binary_classification_metrics_nolabels(outs,preds,kins=kins,get_plots=True,kin_names=kin_names,kin_labels=kin_labels)
    massfit_metrics = classification.get_lambda_mass_fit(kins,true_labels=None,mass_index=-1)#TODO: CHECK THAT ACTUALLY WANT THIS HERE

    # Log to wandb
    if use_wandb:
        wandb.log({
            **metrics,
            **plots,
            **massfit_metrics,
        })

    return massfit_metrics

def experiment(config,use_wandb=True,wandb_project='project',wandb_config={},**kwargs):
    """
    :param: config
    :param: use_wandb
    :param: wandb_project
    :param: wandb_config
    :param: **kwargs

    :return: test_val, apply_val
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
    log_dir          = osp.abspath(config['log_dir'])

    # Log experiment config
    run = None
    if use_wandb: 
        run = wandb.init(project=wandb_project,config=wandb_config,**kwargs)
        wandb.watch(model)

    # Train model
    train(
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

    # Save model
    path = osp.join(log_dir,'model.pt')
    torch.save(model,path)

    # Test model
    test_val  = test(
        model=model,
        device=device,
        dataloader=test_dataloader,
        criterion=criterion,
        kin_names=kin_names,
        kin_labels=kin_labels,
        use_wandb=use_wandb
    )

    # Apply model
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

    return test_val, apply_val
