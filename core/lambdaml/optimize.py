# OPTIMIZE
# pylint: disable=no-member
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.samplers import (
    GridSampler,
    RandomSampler,
    TPESampler,
    CmaEsSampler,
    GPSampler,
    PartialFixedSampler,
    NSGAIISampler,
    QMCSampler,
)
from optuna.pruners import (
    MedianPruner,
    NopPruner,
    PatientPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
    HyperbandPruner,
    ThresholdPruner,
    WilcoxonPruner,
)
import argparse
import wandb
import os
from uuid import uuid4

# Local imports
from .pipeline import pipeline_titok
from .log import setup_logger


# Set module logger
logger = setup_logger(__name__)

# Define available samplers
sampler_choices = {
    "grid": GridSampler,
    "random": RandomSampler,
    "tpe": TPESampler,
    "cmaes": CmaEsSampler,
    "gp": GPSampler,
    "partial_fixed": PartialFixedSampler,
    "nsga2": NSGAIISampler,
    "qmc": QMCSampler,
}

# Define available pruners
pruner_choices = {
    "median": MedianPruner,
    "noprune": NopPruner,
    "patient": PatientPruner,
    "percentile": PercentilePruner,
    "successive_halving": SuccessiveHalvingPruner,
    "hyperband": HyperbandPruner,
    "threshold": ThresholdPruner,
    "wilcoxon": WilcoxonPruner,
}


def parse_suggestion_rule(s):
    """
    Parse suggestion rule string of format:
    - int:start:end
    - float:start:end[:log][:step]
    - cat:val1,val2,...
    """
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"Suggestion must be in format name=rule, got '{s}'"
        )

    name, rule = s.split("=", 1)

    if rule.startswith("int:"):
        parts = rule[4:].split(":")
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(f"Invalid int rule: {rule}")
        start, end = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        return {name: {"type": "int", "range": (start, end), "step": step}}

    elif rule.startswith("float:"):
        parts = rule[6:].split(":")
        if len(parts) < 2:
            raise argparse.ArgumentTypeError(f"Invalid float rule: {rule}")

        low = float(parts[0])
        high = float(parts[1])
        log = False
        step = None

        if len(parts) >= 3 and parts[2]:
            if parts[2] == "log":
                log = True
            else:
                raise argparse.ArgumentTypeError(
                    f"Expected 'log' or empty third field in float rule: {rule}"
                )

        if len(parts) == 4 and parts[3]:
            try:
                step = float(parts[3])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid step value in float rule: {parts[3]}"
                ) from exc

        return {name: {"type": "float", "range": (low, high), "log": log, "step": step}}

    elif rule.startswith("cat:"):
        choices = rule[4:].split(",")
        if not choices:
            raise argparse.ArgumentTypeError(
                f"No choices provided for categorical: {rule}"
            )
        return {name: {"type": "cat", "values": choices}}

    else:
        raise argparse.ArgumentTypeError(f"Unknown rule type: {rule}")


# Define the objective function
def objective(
    trial,
    wandb_project="wandb_project",
    metric_name="auc",
    metric_fn=lambda logs: logs["auc"],
    suggestion_rules=None,
    pipeline=pipeline_titok,
    pipeline_kwargs=None,
):

    # ----- Sample hyperparameters -----#
    if suggestion_rules is None:
        suggestion_rules = {}
    if pipeline_kwargs is None:
        pipeline_kwargs = {}
    suggestions = {"trial": trial}

    # Suggestion rule format:
    # suggestion_rules = {
    #     "param_name": {
    #         "type": "float" | "int" | "categorical",
    #         "range": [min, max],  # for float and int
    #         "values": [val1, val2, ...],  # for categorical
    #         "log": True | False,  # for float
    #         "step": step_size,  # for float
    #     },

    # Loop suggestion rules
    for key in suggestion_rules:
        suggestion_rule = suggestion_rules[key]
        suggestion = None

        # Check that the type is specified
        if type(suggestion_rule) == dict and "type" in suggestion_rule:
            suggestion_type = suggestion_rule["type"]

            # Check if this is a ranged suggestion
            if (
                "range" in suggestion_rule
                and type(suggestion_rule["range"]) in (list, tuple)
                and len(suggestion_rule["range"]) == 2
            ):
                suggestion_range = suggestion_rule["range"]

                # Sample based on type
                if suggestion_type == "float":
                    log = "log" in suggestion_rule and suggestion_rule["log"]
                    step = (
                        suggestion_rule["step"] if "step" in suggestion_rule else None
                    )
                    trial.suggest_float(key, *suggestion_range, step=step, log=log)
                elif suggestion_type == "int":
                    trial.suggest_int(key, *suggestion_range)

            # Check if this is a ranged suggestion
            elif "values" in suggestion_rule and type(suggestion_rule["values"]) in (
                list,
                tuple,
            ):
                suggestion_values = suggestion_rule["values"]

                # Sample based on type
                if suggestion_type == "categorical":
                    trial.suggest_categorial(key, suggestion_values)

        # Error out if you don't recognize the rule
        else:
            raise TypeError("Suggestion rule format not recognized : ", suggestion_rule)

        # Add suggested hyperparameter value
        if suggestion is not None:
            suggestions[key] = suggestion

    # Create a unique output directory for this trial
    experiment_dir = (
        "experiments"
        if "output_dir" not in pipeline_kwargs
        else pipeline_kwargs["output_dir"]
    )
    trial_id = str(uuid4())
    output_dir = os.path.join(experiment_dir, trial_id)
    os.makedirs(output_dir, exist_ok=True)
    suggestions["output_dir"] = output_dir
    trial.set_user_attr("output_dir", output_dir)
    trial.set_user_attr("trial_id", trial_id)

    # Log to wandb
    wandb_run = wandb.init(
        project=wandb_project,
        name=f"trial-{trial.number}",
        config=suggestions,
        dir=str(output_dir),
        reinit=False,
    )

    # Run the pipeline which is assumed to return some logs
    try:
        pipeline_kwargs_updated = pipeline_kwargs.copy()
        pipeline_kwargs_updated.update(suggestions)
        logs = pipeline(**pipeline_kwargs_updated)
    except Exception as exc:
        wandb_run.finish(exit_code=1)
        raise optuna.exceptions.TrialPruned() from exc  # or fail silently

    # Get the AUC from first log dictionary
    metric = metric_fn(logs)

    # Log metrics to wandb
    wandb_run.log({metric_name: metric})
    wandb_run.finish()

    return metric  # Higher is better (maximize)


def optimize(
    storage_url="sqlite:///optuna_study.db",
    optuna_study_direction="maximize",
    optuna_study_name="model_hpo",
    metric_name="auc",
    metric_fn=lambda logs: logs[0]["auc"],
    suggestion_rules=None,
    pipeline=pipeline_titok,
    pipeline_kwargs=None,
    n_trials=100,
    sampler_name="tpe",
    sampler_args=None,
    sampler_kwargs=None,
    pruner_name="median",
    pruner_args=None,
    pruner_kwargs=None,
):

    # Check arguments
    if not callable(metric_fn):
        raise TypeError("metric_fn must be callable")
    if suggestion_rules is None:
        suggestion_rules = {}
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    # Create database
    storage = optuna.storages.RDBStorage(
        url=storage_url  # or your PostgreSQL/MySQL URL
    )

    # Create sampler
    sampler = (
        sampler_choices[sampler_name.lower()](
            *([] if sampler_args is None else sampler_args),
            **({} if sampler_kwargs is None else sampler_kwargs),
        )
        if sampler_name.lower() in sampler_choices
        else None
    )

    # Create pruner
    pruner = (
        pruner_choices[pruner_name.lower()](
            *([] if pruner_args is None else pruner_args),
            **({} if pruner_kwargs is None else pruner_kwargs),
        )
        if pruner_name.lower() in pruner_choices
        else None
    )

    # Create study
    study = optuna.create_study(
        direction=optuna_study_direction,
        study_name=optuna_study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # Optimize
    wandb_callback = WeightsAndBiasesCallback(metric_name=metric_name, as_multirun=True)
    callbacks = [wandb_callback]
    study.optimize(
        lambda trial: objective(
            trial,
            metric_name=metric_name,
            metric_fn=metric_fn,
            suggestion_rules=suggestion_rules,
            pipeline=pipeline,
            pipeline_kwargs=pipeline_kwargs,
        ),
        n_trials=n_trials,
        callbacks=callbacks,
    )
