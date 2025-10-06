# MODELS
# pylint: disable=no-member
# pylint: disable=abstract-method
# pylint: disable=arguments-differ
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    global_mean_pool,
)

# Local imports
from .log import setup_logger


# Set module logger
logger = setup_logger(__name__)


# Gradient Reversal Function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        x, alpha = args
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        return -ctx.alpha * grad_output, None


class FlexibleGNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hdim,
        num_layers=2,
        gnn_type="gcn",  # Options: 'gcn', 'sage', 'gat', 'gin'
        dropout=0.5,
        heads=1,  # For GAT
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()

        # First layer
        self.convs.append(self._build_conv(self.gnn_type, in_dim, hdim, heads))
        self.bns.append(
            nn.BatchNorm1d(hdim * (heads if self.gnn_type == "gat" else 1))
        )

        # Hidden layers
        for _ in range(num_layers - 1):
            in_ch = hdim * (heads if self.gnn_type == "gat" else 1)
            out_ch = hdim
            self.convs.append(self._build_conv(self.gnn_type, in_ch, out_ch, heads))
            self.bns.append(
                nn.BatchNorm1d(out_ch * (heads if self.gnn_type == "gat" else 1))
            )

    def _build_conv(self, gnn_type, in_dim, out_dim, heads):
        if gnn_type == "gcn":
            return GCNConv(in_dim, out_dim)
        elif gnn_type == "sage":
            return SAGEConv(in_dim, out_dim)
        elif gnn_type == "gat":
            return GATConv(in_dim, out_dim, heads=heads, concat=True)
        elif gnn_type == "gin":
            return GINConv(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
                )
            )
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class GraphClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hdim=64, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        # Input layer
        if num_layers == 1:
            self.layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.layers.append(nn.Linear(in_dim, hdim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hdim, hdim))
            self.layers.append(nn.Linear(hdim, out_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, num_layers=2, hdim=64, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.layers.append(nn.Linear(in_dim, 2))
        else:
            self.layers.append(nn.Linear(in_dim, hdim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hdim, hdim))
            self.layers.append(nn.Linear(hdim, 2))  # binary domain classification

    def forward(self, x, alpha=1.0):
        x = GradReverse.apply(x, alpha)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# Projection head for contrastive loss
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=64, out_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.proj(x)
