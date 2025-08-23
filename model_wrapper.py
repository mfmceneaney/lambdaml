# model_wrapper.py
import torch
from torch_geometric.data import Data
from models import *

class ModelWrapper:
    def __init__(self, path='model_checkpoint.pt'):

        # TODO: Load both encoder and classifier

        self.model = GCN()
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()

    def predict(self, graph_json):
        x = torch.tensor(graph_json['x'], dtype=torch.float)
        edge_index = torch.tensor(graph_json['edge_index'], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            logits = self.model(data)
            prob = torch.sigmoid(logits).item()

        return prob
