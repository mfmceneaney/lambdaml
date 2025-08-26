#----------------------------------------------------------------------------------------------------#
# OPTIMIZE
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
import os
from uuid import uuid4
import sqlite3

# Local imports
from pipeline import pipeline_titok

# Define the objective function
def objective(trial, metric_name="auc", metric_fn=lambda logs: return logs["auc"], suggestion_rules={}, pipeline=pipeline_titok, pipeline_kwargs={}):

    #----- Sample hyperparameters -----#
    suggestions = {"trial":trial}

    # Loop suggestion rules
    for key in suggestion_rules:
        suggestion_rule = suggestion_rules[key]
        suggestion = None

        # Check that the type is specified
        if type(suggestion_rule)==dict and "type" in suggestion_rule:
            suggestion_type = suggestion_rule["type"]

            # Check if this is a ranged suggestion
            if "range" in suggestion_rule and \
            type(suggestion_rule["range"]) in (list,tuple) and len(suggestion_rule["range"])==2:
                suggestion_range = suggestion_rule["range"]

                # Sample based on type
                if suggestion_type == "float":
                    log = "log" in suggestion_rule and suggestion_rule["log"]
                    step = suggestion_rule["step"] if "step" in suggestion_rule else 1
                    trial.suggest_float(key, *suggestion_range, step=step, log=log)
                elif suggestion_type == "int":
                    trial.suggest_int(key, *suggestion_range)

            # Check if this is a ranged suggestion
            elif "values" in suggestion_rule and \
            type(suggestion_rule["values"]) in (list,tuple):
                suggestion_value = suggestion_rule["values"]

                # Sample based on type
                if suggestion_type == "categorical":
                    trial.suggest_categorial(key, suggestion_values)

        # Error out if you don't recognize the rule
        else:
            raise TypeError("Suggestion rule format not recognized : ",suggestion_rule)

        # Add suggested hyperparameter value
        if suggestion is not None:
            suggestions[key] = suggestion

    # Create a unique output directory for this trial
    experiment_dir = "experiments" if "output_dir" not in pipeline_kwargs
    trial_id = str(uuid4())
    output_dir = osp.join(experiment_dir,trial_id)
    osp.makedirs(output_dir exist_ok=True)
    suggestions["output_dir"] = output_dir
    trial.set_user_attr("output_dir", output_dir)
    trial.set_user_attr("trial_id", trial_id)

    # Log to wandb
    wandb_run = wandb.init(
        project=wandb_project,
        name=f"trial-{trial.number}",
        config=suggestions,
        dir=str(output_dir),
        reinit=True,
    )

    # Run the pipeline which is assumed to return some logs
    try:
        pipeline_kwargs_updated = pipeline_kwargs.copy()
        pipeline_kwargs_updated.update(suggestions)
        logs = pipeline(**pipeline_kwargs_updated)
    except Exception as e:
        wandb_run.finish(exit_code=1)
        raise optuna.exceptions.TrialPruned()  # or fail silently

    # Get the AUC from first log dictionary
    metric = metric_fn(logs)

    # Log metrics to wandb
    wandb_run.log({metric_name: metric})
    wandb_run.finish()

    return metric  # Higher is better (maximize)

def optimize(
        storage_url = "sqlite:///optuna_study.db",
        optuna_study_direction="maximize",
        optuna_study_name="model_hpo",
        metric_name = "auc",
        metric_fn = lambda logs: return logs[0]["auc"],
        suggestion_rules = {},
        pipeline_kwargs = {},
        n_trials = 100
    ):

    # Create database
    storage = optuna.storages.RDBStorage(
        url=storage_url  # or your PostgreSQL/MySQL URL
    )

    # Create study
    study = optuna.create_study(
        direction=optuna_study_direction,
        study_name=optuna_study_name,
        storage=storage,
        load_if_exists=True,
    )

    # Optimize
    wandb_callback = WeightsAndBiasesCallback(metric_name=metric_name, as_multirun=True)
    callbacks=[wandb_callback]
    study.optimize(
            lambda trial: objective(
                trial,
                metric_name=metric_name,
                metric_fn=metric_fn,
                suggestion_rules=suggestion_rules,
                pipeline=pipeline_titok,
                pipeline_kwargs=pipeline_kwargs
            ),
            n_trials=n_trials,
            callbacks=callbacks
        )
