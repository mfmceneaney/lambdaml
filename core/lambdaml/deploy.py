# DEPLOY
# pylint: disable=no-member
import torch
from torch_geometric.data import Data
import json
import optuna
import shutil
import os
import os.path as osp
from codename import codename

# Local imports
from .models import FlexibleGNNEncoder, GraphClassifier
from .preprocess import preprocess_rec_particle
from .log import setup_logger
from .util import save_json


# Set module logger
logger = setup_logger(__name__)


class ModelWrapper:
    def __init__(
        self,
        encoder_type=FlexibleGNNEncoder,
        trial_dir="app/registry/model_hpo/trial_id",
        encoder_fname="encoder.pt",
        encoder_params_fname="encoder_params.json",
        clf_type=GraphClassifier,
        clf_fname="clf.pt",
        clf_params_fname="clf_params.json",
        map_location="cpu",
        preprocessing_fn=preprocess_rec_particle,
        preprocessing_fn_kwargs=None,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    ):

        # Set attributes
        logger.debug("Using trial directory: %s", trial_dir)
        self.trial_dir = osp.abspath(trial_dir)
        logger.debug("Using device: %s", device_name)
        self.device = torch.device(device_name)
        self.preprocessing_fn = preprocessing_fn
        self.preprocessing_fn_kwargs = (
            preprocessing_fn_kwargs if preprocessing_fn_kwargs is not None else {}
        )

        # Load encoder
        encoder_params_path = osp.join(self.trial_dir, encoder_params_fname)
        logger.debug("Loading encoder parameters from %s", encoder_params_path)
        encoder_params = {}
        with open(encoder_params_path, "r", encoding="utf-8") as f:
            encoder_params = json.load(f)
        logger.debug("Creating encoder from encoder_params = %s", encoder_params)
        self.encoder = encoder_type(**encoder_params)
        encoder_path = osp.join(self.trial_dir, encoder_fname)
        logger.debug(
            "Loading encoder state dictionary to %s from %s",
            map_location,
            encoder_path,
        )
        self.encoder.load_state_dict(
            torch.load(encoder_fname, map_location=map_location)
        )
        self.encoder.eval()

        # Load classifier
        clf_params_path = osp.join(self.trial_dir, clf_params_fname)
        logger.debug("Loading classifier parameters from %s", clf_params_path)
        clf_params = {}
        with open(clf_params_path, "r", encoding="utf-8") as f:
            clf_params = json.load(f)
        logger.debug("Creating classifier from clf_params = %s", clf_params)
        self.clf = clf_type(**clf_params)
        clf_path = osp.join(self.trial_dir, clf_fname)
        logger.debug(
            "Loading classifier state dictionary to %s from %s", map_location, clf_path
        )
        self.clf.load_state_dict(torch.load(clf_fname, map_location=map_location))
        self.clf.eval()

        # Copy models to device
        logger.debug("Copying encoder and classifier to device = %s", self.device)
        self.encoder = self.encoder.to(self.device)
        self.clf = self.clf.to(self.device)

    def predict(self, bank_tables):

        # Preprocess graph from an event tables dictionary or json
        logger.debug("Loading data from bank_tables = %s", bank_tables)
        x, edge_index = None, None
        if callable(self.preprocessing_fn):
            logger.debug(
                "Obtaining x, edge_index from preprocessing function with kwargs %s",
                self.preprocessing_fn_kwargs,
            )
            x, edge_index = self.preprocessing_fn(
                bank_tables, **self.preprocessing_fn_kwargs
            )
            logger.debug("Copying x, edge_index tensors to device = %s", self.device)
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)

        else:
            raise TypeError(
                "Preprocessing function must be calllable, but found type(self.preprocessing_fn) =",
                type(self.preprocessing_fn),
            )

        # Create the graph
        logger.debug("Creating graph from x = %s\n\tand edge_index = %s", x, edge_index)
        data = Data(x=x, edge_index=edge_index)

        # Run the graph through the model
        logger.debug("Applying encoder, classifier, softmax")
        with torch.no_grad():
            feat = self.encoder(data)
            logit = self.clf(feat)
            prob = torch.softmax(logit, dim=1).item()

        return prob


def select_best_models(
    n_best_trials=1,
    optuna_study_name="gnn_study",
    optuna_storage_url="sqlite:///optuna.db",
    registry="app/registry/",
    codename_separator="-",
    encoder_fname="encoder.pt",
    encoder_params_fname="encoder_params.json",
    clf_fname="clf.pt",
    clf_params_fname="clf_params.json",
):

    # Connect to the study in the SQL DB
    study = optuna.load_study(
        study_name=optuna_study_name,
        storage=optuna_storage_url,  # or use your DB URI
    )

    # Get all completed trials, sorted by value
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # Sort by objective value (ascending for minimization)
    top_n_trials = sorted(completed_trials, key=lambda t: t.value)[:n_best_trials]

    # Record top n trials ordering and codenames to trials map
    trials_to_codenames = {
        t.number: codename(id=str(t.number), separator=codename_separator)
        for t in top_n_trials
    }
    metadata_dir = osp.join(osp.abspath(registry), optuna_study_name)
    logger.debug("Creating app study directory %s", metadata_dir)
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = osp.join(osp.abspath(registry), optuna_study_name, "metadata.json")
    logger.debug("Saving app metadata to %s", metadata_path)
    save_json(metadata_path, trials_to_codenames)

    # Loop n trials
    for i, trial in enumerate(top_n_trials):

        # Access hyperparameters and custom user attributes
        logger.info("Trial rank: %d", i)
        logger.info("Trial number: %s", trial.number)
        logger.info("Trial codename: %s", trials_to_codenames[trial.number])
        logger.info("Trial value: %s", trial.value)
        logger.info("Hyperparameters: %s", trial.params)
        logger.info("Trial user attributes: %s", trial.user_attrs)

        # Load encoder and classifier file paths (assuming you stored them with trial.set_user_attr)
        encoder_path = trial.user_attrs["encoder_path"]
        encoder_params_path = trial.user_attrs["encoder_params_path"]
        clf_path = trial.user_attrs["clf_path"]
        clf_params_path = trial.user_attrs["clf_params_path"]

        # Check that encoder and classifier paths are full paths
        trial_out_dir = trial.user_attrs["out_dir"]
        if not encoder_path.startswith(osp.sep):
            encoder_path = osp.abspath(osp.join(trial_out_dir, encoder_path))
        if not encoder_params_path.startswith(osp.sep):
            encoder_params_path = osp.abspath(osp.join(trial_out_dir, encoder_params_path))
        if not clf_path.startswith(osp.sep):
            clf_path = osp.abspath(osp.join(trial_out_dir, clf_path))
        if not clf_params_path.startswith(osp.sep):
            clf_params_path = osp.abspath(osp.join(trial_out_dir, clf_params_path))

        # Set trial application directory
        trial_registry = osp.abspath(
            osp.join(osp.abspath(registry), optuna_study_name, trials_to_codenames[trial.number])
        )
        logger.debug("Creating trial registry directory %s", trial_registry)
        os.makedirs(trial_registry, exist_ok=True)
        logger.debug("Copying trial %s to %s", trial.number, trial_registry)

        # Copy models and params to gnn server directory
        logger.debug(
            "Copying encoder:\n\t%s\n\t -> %s",
            encoder_path,
            osp.join(trial_registry, encoder_fname),
        )
        shutil.copy(encoder_path, osp.join(trial_registry, encoder_fname))
        logger.debug(
            "Copying encoder params:\n\t%s\n\t -> %s",
            encoder_params_path,
            osp.join(trial_registry, osp.basename(encoder_params_fname)),
        )
        shutil.copy(
            encoder_params_path,
            osp.join(trial_registry, osp.basename(encoder_params_fname)),
        )
        logger.debug(
            "Copying classifier:\n\t%s\n\t -> %s",
            clf_path,
            osp.join(trial_registry, clf_fname),
        )
        shutil.copy(clf_path, osp.join(trial_registry, osp.basename(clf_fname)))
        logger.debug(
            "Copying classifier params:\n\t%s\n\t -> %s",
            clf_params_path,
            osp.join(trial_registry, osp.basename(clf_params_fname)),
        )
        shutil.copy(
            clf_params_path, osp.join(trial_registry, osp.basename(clf_params_fname))
        )
