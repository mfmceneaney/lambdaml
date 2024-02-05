#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

import torch
import torch_geometric
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import MLP, GCNConv, GINConv, GraphConv, global_add_pool, global_mean_pool,  global_max_pool
from torch_geometric.nn.norm import GraphNorm, BatchNorm

class GCN(torch.nn.Module):
    def __init__(
                    self,
                    in_channels=7,
                    gnn_num_layers=3,
                    gnn_conv='GCNConv',
                    gnn_norm='GraphNorm',
                    head_num_mlp_layers=3,
                    head_mlp_hidden_dim=128,
                    head_norm=None,
                    head_act='relu',
                    dropout=0.5,
                    out_channels=2
                ):
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

        # Set parameters
        self.in_channels = in_channels
        self.gnn_num_layers = gnn_num_layers
        self.gnn_conv = gnn_conv
        self.conv = GCNConv if self.gnn_conv == 'GCNConv' else GraphConv
        self.gnn_norm = gnn_norm
        self.norm = GraphNorm if self.gnn_conv == 'GraphNorm' else BatchNorm
        self.head_num_mlp_layers = head_num_mlp_layers
        self.head_mlp_hidden_dim = head_mlp_hidden_dim
        self.head_norm = head_norm
        self.head_act = head_act
        self.dropout = dropout
        self.out_channels = out_channels

        # Set GNN and normalization layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for idx in range(self.gnn_num_layers - 1):
            layers = [self.gnn_mlp_hidden_dim for i in range(2)]
            if idx==0:
                layers[0] = self.in_channels
            self.convs.append(self.conv(*layers))
            self.norms.append(self.norm(layers[-1]))

        # Set head MLP layers
        mlp_layers = [self.head_mlp_hidden_dim for i in range(self.head_num_mlp_layers-2)]
        mlp_layers.append(out_channels)
        mlp_layers.insert(0,self.gnn_mlp_hidden_dim)
        self.head_mlp = MLP(mlp_layers, norm=self.head_norm, dropout=self.dropout, act=self.head_act)

    def forward(self, x, edge_index, batch):

        # Apply convolutional layers
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = self.gnn_act(x)

        # Readout layer
        x = self.pool(x, batch)  # [batch_size, hidden_channels]

        # Head classifier
        x = self.head_mlp(x)

        return x

class GIN(torch.nn.Module):
    def __init__(
                    self,
                    in_channels = 7,
                    gnn_num_layers = 4,
                    gnn_num_mlp_layers = 3,
                    gnn_mlp_hidden_dim = 128,
                    gnn_mlp_norm = 'batch_norm',
                    gnn_mlp_act = 'relu',
                    train_epsilon = False,
                    head_num_mlp_layers = 3,
                    head_mlp_hidden_dim =  128,
                    head_norm = None,
                    head_act = 'relu',
                    dropout = 0.5,
                    out_channels = 2,
                    pool = 'max'
                ):
        super().__init__()

        # Set parameters
        self.in_channels = in_channels
        self.gnn_num_layers = gnn_num_layers
        self.gnn_num_mlp_layers = gnn_num_mlp_layers
        self.gnn_mlp_hidden_dim = gnn_mlp_hidden_dim
        self.gnn_mlp_norm = gnn_mlp_norm
        self.gnn_mlp_act = gnn_mlp_act
        self.train_epsilon = train_epsilon
        self.head_num_mlp_layers = head_num_mlp_layers
        self.head_mlp_hidden_dim = head_mlp_hidden_dim
        self.head_norm = head_norm
        self.head_act = head_act
        self.dropout = dropout
        self.out_channels = out_channels
        self.pool = global_add_pool if pool=='sum' else global_mean_pool if pool=='mean' else global_max_pool

        # Set GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for idx in range(self.gnn_num_layers - 1):
            mlp_layers = [self.gnn_mlp_hidden_dim for i in range(self.gnn_num_mlp_layers-2)]
            mlp_layers.append(out_channels)
            if idx==0:
                mlp_layers.insert(0,self.in_channels)
            else:
                mlp_layers.insert(0,self.gnn_mlp_hidden_dim)
            mlp = MLP(mlp_layers, norm=self.gnn_mlp_norm, act=self.gnn_mlp_act)
            self.convs.append(GINConv(mlp, train_eps=self.train_epsilon))
            self.bns.append(BatchNorm1d(hidden_dim))

        # Set MLP head layers, one mlp for each GNN layer
        self.mlps = torch.nn.ModuleList()
        for idx in range(self.gnn_num_layers): #NOTE: INPUT DIM FOR MLP LAYER HAS TO MATCH INPUT DIM FOR EACH GRAPH REPRESENTATION IN EACH LAYER HERE
            mlp_layers = [self.head_mlp_hidden_dim for i in range(self.head_num_mlp_layers-2)]
            mlp_layers.append(out_channels)
            if idx==0:
                mlp_layers.insert(0,self.in_channels)
            else:
                mlp_layers.insert(0,self.head_mlp_hidden_dim)
            self.mlps.append(MLP(mlp_layers,norm=self.head_norm, dropout=self.dropout, act=self.head_act))

    def forward(self, x, edge_index, batch):
        hidden_rep = [x]
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            hidden_rep.append(x)

        # Aggregate over graph representations from different GNN layers
        score_over_layer = 0
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(h,batch)
            newval = self.mlps[i](pooled_h)
            score_over_layer += newval#self.mlps[i](pooled_h)

        return score_over_layer

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

class ParticleNet(torch.nn.Module):

    def __init__(
        self,
        settings = {
                "conv_params": [
                    (2, (64, 64, 256)),
                    (16, (64, 64, 64)),
                    (16, (128, 128, 128)),
                    (16, (256, 256, 256)),
                ],
                "fc_params": [
                    (0.1, 256)
                ],
                "input_features": 7,
                "output_classes": 2,
                "device":"cpu"
            }
        ):
        super().__init__()
        previous_output_shape = settings['input_features']

        self.input_bn = torch_geometric.nn.BatchNorm(settings['input_features'])

        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['conv_params']):
            K, channels = layer_param
            self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=K).to(settings['device']))#NOTE: Originally : .to(DEVICE)
            previous_output_shape = channels[-1]

        self.fc_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['fc_params']):
            drop_rate, units = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p=drop_rate),
                torch.nn.ReLU()
            ).to(settings['device'])#NOTE: Originally : .to(DEVICE)
            self.fc_process.append(seq)
            previous_output_shape = units


        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, settings['output_classes'])
        # self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        fts = self.input_bn(batch.x)
        pts = batch.pos

        for idx, layer in enumerate(self.conv_process):
            fts = layer(pts, fts, batch.batch)
            pts = fts

        x = torch_geometric.nn.global_mean_pool(fts, batch.batch)

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)
        # x = self.output_activation(x)
        return x
