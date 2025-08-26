# ----------------------------------------------------------------------------------------------------#
# DEPLOY
import torch
from torch_geometric.data import Data
from models import *
import json
import optuna
import shutil


class TITOKModelWrapper:
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
        encoder_params = {}
        with open(encoder_params_path, "r") as f:
            encoder_params = json.load(f)
        self.encoder = type_encoder(**encoder_params)
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=map_location)
        )
        self.encoder.eval()

        # Load classifier
        clf_params = {}
        with open(clf_params_path, "r") as f:
            clf_params = json.load(f)
        self.clf = type_clf(**clf_params)
        self.clf.load_state_dict(torch.load(clf_path, map_location=map_location))
        self.clf.eval()

    def predict(self, graph_json):

        # Assume you are passing a json or dictionary object
        x = torch.tensor(graph_json["x"], dtype=torch.float)
        edge_index = torch.tensor(graph_json["edge_index"], dtype=torch.long)

        # Create the graph
        data = Data(x=x, edge_index=edge_index)

        # Run the graph through the model
        with torch.no_grad():
            feat = self.encoder(data)
            logit = self.clf(feat)
            prob = torch.softmax(logits, dim=1).item()

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
        study_name="gnn_study", storage="sqlite:///optuna.db"  # or use your DB URI
    )

    # Get the best trial
    best_trial = study.best_trial

    # Access hyperparameters and custom user attributes
    print("Best trial value:", best_trial.value)
    print("Hyperparameters:", best_trial.params)
    print("Trial user attributes:", best_trial.user_attrs)

    # Load encoder and classifier file paths (assuming you stored them with trial.set_user_attr)
    encoder_path = best_trial.user_attrs["encoder_path"]
    encoder_params_path = best_trial.user_attrs["encoder_params_path"]
    clf_path = best_trial.user_attrs["clf_path"]
    clf_params_path = best_trial.user_attrs["clf_params_path"]

    # Copy models and params to gnn server directory
    shutil.copy(encoder_path, osp.join(gnn_server_dir, encoder_fname))
    shutil.copy(
        encoder_params_path,
        osp.join(gnn_server_dir, osp.basename(encoder_params_fname)),
    )
    shutil.copy(clf_path, osp.join(gnn_server_dir, osp.basename(clf_fname)))
    shutil.copy(
        clf_params_path, osp.join(gnn_server_dir, osp.basename(clf_params_fname))
    )
