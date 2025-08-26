#----------------------------------------------------------------------------------------------------#
# MODELS
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
import pytest

# Local imports
from core.models import *

# TODO