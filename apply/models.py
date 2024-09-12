#TODO: CREATE MODEL
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.norm import GraphNorm, BatchNorm

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)#.jittable() #NOTE: NEEDED FOR DEPLOYMENT IN CMAKE
        self.conv2 = GCNConv(hidden_channels, hidden_channels)#.jittable()
        self.conv3 = GCNConv(hidden_channels, hidden_channels)#.jittable()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
        self.bn1 = torch_geometric.nn.norm.GraphNorm(hidden_channels)
        self.bn2 = torch_geometric.nn.norm.GraphNorm(hidden_channels)
        self.bn3 = torch_geometric.nn.norm.GraphNorm(hidden_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)

        return x

import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GINConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.norm import GraphNorm
from torch.nn import BatchNorm1d

class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        hidden_dim = 128 #hidden_dim
        num_layers = 8 # num_layers
        self.num_layers = num_layers
        
        for _ in range(self.num_layers - 1):
            mlp = MLP([in_channels if _==0 else hidden_dim, hidden_dim, hidden_dim, hidden_dim],norm='batch_norm',act='relu') #NOTE ADDED EXTRA LAYER HERE
            self.convs.append(GINConv(mlp, train_eps=False))
            self.bns.append(BatchNorm1d(hidden_dim))

        self.mlps = torch.nn.ModuleList()
        for _ in range(self.num_layers): #NOTE: INPUT DIM FOR MLP LAYER HAS TO MATCH INPUT DIM FOR EACH GRAPH REPRESENTATION IN EACH LAYER HERE
            self.mlps.append(MLP([in_channels if _==0 else hidden_dim, hidden_dim, hidden_dim, out_channels], norm=None, dropout=0.0, act='relu')) #NOTE ADDED EXTRA LAYER HERE AND..-> CHANGED ACTIVATION FUNCTION FROM DEFAULT RELU

    def forward(self, x, edge_index, batch): #data):
        #x, edge_index, batch = data.x, data.edge_index, data.batch
        hidden_rep = [x]
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            hidden_rep.append(x)
        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
#             pooled_h = global_add_pool(h,batch)#NOTE: ORIGINAL
            pooled_h = global_max_pool(h,batch)
            newval = self.mlps[i](pooled_h)
            score_over_layer += newval#self.mlps[i](pooled_h)
        return score_over_layer

'''
    ParticleNet Implementation
'''

class ParticleStaticEdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ParticleStaticEdgeConv, self).__init__(aggr='max')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, out_channels[0], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[0]), 
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[0], out_channels[1], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[1], out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index, k):
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, edge_index, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)

        out_mlp = self.mlp(tmp)

        return out_mlp

    def update(self, aggr_out):
        return aggr_out

class ParticleDynamicEdgeConv(ParticleStaticEdgeConv):
    def __init__(self, in_channels, out_channels, k=7):
        super(ParticleDynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k
        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()

    def forward(self, pts, fts, batch=None):
        edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False, flow=self.flow)
        aggrg = super(ParticleDynamicEdgeConv, self).forward(fts, edges, self.k)
        x = self.skip_mlp(fts)
        out = torch.add(aggrg, x)
        return self.act(out)


# class ParticleNet(torch.nn.Module):

#     def __init__(self, settings):
#         super().__init__()
#         previous_output_shape = settings['input_features']

#         self.input_bn = torch_geometric.nn.BatchNorm(settings['input_features'])

#         self.conv_process = torch.nn.ModuleList()
#         for layer_idx, layer_param in enumerate(settings['conv_params']):
#             K, channels = layer_param
#             self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=K).to(settings['device']))#NOTE: Originally : .to(DEVICE)
#             previous_output_shape = channels[-1]



#         self.fc_process = torch.nn.ModuleList()
#         for layer_idx, layer_param in enumerate(settings['fc_params']):
#             drop_rate, units = layer_param
#             seq = torch.nn.Sequential(
#                 torch.nn.Linear(previous_output_shape, units),
#                 torch.nn.Dropout(p=drop_rate),
#                 torch.nn.ReLU()
#             ).to(settings['device'])#NOTE: Originally : .to(DEVICE)
#             self.fc_process.append(seq)
#             previous_output_shape = units


#         self.output_mlp_linear = torch.nn.Linear(previous_output_shape, settings['output_classes'])
#         self.output_activation = torch.nn.Softmax(dim=1)

#     def forward(self, batch):
#         fts = self.input_bn(batch.x)
#         pts = batch.pos

#         for idx, layer in enumerate(self.conv_process):
#             fts = layer(pts, fts, batch.batch)
#             pts = fts

#         x = torch_geometric.nn.global_mean_pool(fts, batch.batch)

#         for layer in self.fc_process:
#             x = layer(x)

#         x = self.output_mlp_linear(x)
# #         x = self.output_activation(x)
#         return x

# settings = {
#     "conv_params": [
#         (2, (64, 64, 64)),
#         (2, (64, 64, 64)),
# #         (2, (64, 64, 256)),
# #         (16, (64, 64, 64)),
# #         (16, (128, 128, 128)),
# #         (16, (256, 256, 256)),
#     ],
#     "fc_params": [
#         (0.5, 64)
# #         (0.1, 256)
#     ],
#     "input_features": dataset.num_node_features, # default was 4 (e, px, py, pz)
#     "output_classes": dataset.num_classes,       # default was 2
#     "device":device
# }

# # model = ParticleNet(settings)
# # # model = model.to(DEVICE)

# # print(model)
