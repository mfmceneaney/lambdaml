# DEPLOY
# pylint: disable=no-member
import torch
from torch_geometric.data import Data
import json
import optuna
import shutil
import os.path as osp

# Local imports
from .models import FlexibleGNNEncoder, GraphClassifier
from .log import setup_logger


# Set module logger
logger = setup_logger(__name__)


class ModelWrapper:
    def __init__(
        self,
        encoder_type=FlexibleGNNEncoder,
        encoder_path="encoder.pt",
        encoder_params_path="encoder_params.json",
        clf_type=GraphClassifier,
        clf_path="clf.pt",
        clf_params_path="clf_params.json",
        map_location="cpu",
    ):

        # Load encoder
        logger.debug("Loading encoder parameters from %s", encoder_params_path)
        encoder_params = {}
        with open(encoder_params_path, "r", encoding="utf-8") as f:
            encoder_params = json.load(f)
        logger.debug("Creating encoder from encoder_params = %s", encoder_params)
        self.encoder = encoder_type(**encoder_params)
        logger.debug("Loading encoder state dictionary to %s from %s", map_location, encoder_path)
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=map_location)
        )
        self.encoder.eval()

        # Load classifier
        logger.debug("Loading classifier parameters from %s", clf_params_path)
        clf_params = {}
        with open(clf_params_path, "r", encoding="utf-8") as f:
            clf_params = json.load(f)
        logger.debug("Creating classifier from clf_params = %s", clf_params)
        self.clf = clf_type(**clf_params)
        logger.debug("Loading classifier state dictionary to %s from %s", map_location, clf_path)
        self.clf.load_state_dict(torch.load(clf_path, map_location=map_location))
        self.clf.eval()

    def predict(self, graph_json):

        # Assume you are passing a json or dictionary object
        logger.debug("Loading data from graph_json[\"x\",\"edge_index\"]")
        x = torch.tensor(graph_json["x"], dtype=torch.float)
        edge_index = torch.tensor(graph_json["edge_index"], dtype=torch.long)

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


def select_best_model(
    optuna_study_name="gnn_study",
    optuna_storage="sqlite:///optuna.db",
    gnn_server_dir="gnn_server/",
    encoder_fname="encoder.pt",
    encoder_params_fname="encoder_params.json",
    clf_fname="clf.pt",
    clf_params_fname="clf_params.json",
):

    # Connect to the study in the SQL DB
    study = optuna.load_study(
        study_name=optuna_study_name,
        storage=optuna_storage,  # or use your DB URI
    )

    # Get the best trial
    best_trial = study.best_trial

    # Access hyperparameters and custom user attributes
    logger.info("Best trial value: %s", best_trial.value)
    logger.info("Hyperparameters: %s", best_trial.params)
    logger.info("Trial user attributes: %s", best_trial.user_attrs)

    # Load encoder and classifier file paths (assuming you stored them with trial.set_user_attr)
    encoder_path = best_trial.user_attrs["encoder_path"]
    encoder_params_path = best_trial.user_attrs["encoder_params_path"]
    clf_path = best_trial.user_attrs["clf_path"]
    clf_params_path = best_trial.user_attrs["clf_params_path"]

    # Copy models and params to gnn server directory
    logger.debug("Copying encoder:\n\t%s\n\t -> %s", encoder_path, osp.join(gnn_server_dir, encoder_fname))
    shutil.copy(encoder_path, osp.join(gnn_server_dir, encoder_fname))
    logger.debug("Copying encoder params:\n\t%s\n\t -> %s", encoder_params_path, osp.join(gnn_server_dir, osp.basename(encoder_params_fname)))
    shutil.copy(
        encoder_params_path,
        osp.join(gnn_server_dir, osp.basename(encoder_params_fname)),
    )
    logger.debug("Copying classifier:\n\t%s\n\t -> %s", clf_path, osp.join(gnn_server_dir, clf_fname))
    shutil.copy(clf_path, osp.join(gnn_server_dir, osp.basename(clf_fname)))
    logger.debug("Copying classifier params:\n\t%s\n\t -> %s", clf_params_path, osp.join(gnn_server_dir, osp.basename(clf_params_fname)))
    shutil.copy(
        clf_params_path, osp.join(gnn_server_dir, osp.basename(clf_params_fname))
    )
