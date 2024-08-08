#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

# Logging
import wandb

# Optimization
import optuna

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
    torch.save(model,path,map_location='cpu')

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

def optimize(
            config={},
            opt_par_config={},
            use_wandb=True,
            wandb_project='project',
            wandb_config={},
            study_name='study',
            direction='minimize',
            minimization_key='roc_auc',
            load_if_exists=True,
            ntrials=100,
            timeout=864000,
            gc_after_trial=True,
            log_dir="./",
            **kwargs
            ):
    """
    :param: opt_par_config
        :description: Configuration dictionary with structure {opt_par_name:{'lims':[opt_par_min,opt_par_max], 'log':False}}
    :param: use_wandb
    :param: wandb_project
    :param: wandb_config
    :param: config
    :param: study_name
    :param: direction
    :param: minimization_key
    :param: load_if_exists
    :param: ntrials
    :param: timeout
    :param: gc_after_trial
    :param: log_dir
    """

    def objective(trial):

        trial_config = config.copy() #NOTE: COPY IS IMPORTANT HERE!

        # Suggest trial params and substitute into trial_config also set log dir name with trial param of objective
        for idx, opt_par_name in enumerate(opt_par_config):
            lims = opt_par_config[opt_par]['lims']
            if len(lims)!=2:
                raise TypeError("opt_par_config limits must be length 2 but opt_par_config[",opt_par,"]['lims'] has length",len(lims))
            log = opt_par_config[opt_par]['log']
            trial_opt_par = None
            if type(lims[0])==int:
                trial_opt_par = trial.suggest_int(opt_par_name,lims[0],lims[1],log=log)
            elif type(lims[0])==float:
                trial_opt_par = trial.suggest_float(opt_par_name,lims[0],lims[1],log=log)
            trial_config[opt_par] = trial_opt_par

        # Set trial number so this gets passed to wandb
        trial_config['trial_number'] = trial.number

        # Create output directory if it does not exist
        trial_log_dir = os.path.abspath(log_dir)+"/trial_"+str(trial.number)
        if not os.path.isdir(trial_log_dir):
            try:
                os.makedirs(trial_log_dir) #NOTE: Do NOT use os.path.join() here since it requires that the directory exist.
            except Exception:
                if verbose: print("Could not create output directory:",trial_log_dir)

        experiment_val = experiment(
            trial_config,
            use_wandb=use_wandb,
            wandb_project=project,
            wandb_config=wandb_config,
            **kwargs
        )
        optimization_value = experiment_val[0][minimization_key]#NOTE: Get roc_auc from experiment testing values #TODO: Add option for setting index here so can select based on application values

        return 1.0-optimization_value if minimization_key=='roc_auc' else optimization_value #NOTE: #TODO: CHANGE THIS FROM BEING HARD-CODED

    # Load or create pruner, sampler, and study
    pruner = optuna.pruners.MedianPruner() if opt_config['pruning'] else optuna.pruners.NopPruner() #TODO: Add CLI options for other pruners
    sampler = TPESampler() #TODO: Add command line option for selecting different sampler types.
    study = optuna.create_study(
        storage='sqlite:///'+opt_config['db_path'],
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        direction=direction,
        load_if_exists=load_if_exists,
    ) #TODO: Add options for different SQL programs: Postgre, MySQL, etc.

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=gc_after_trial
    ) #NOTE: gc_after_trial=True is to avoid OOM errors see https://optuna.readthedocs.io/en/stable/faq.html#out-of-memory-gc-collect
