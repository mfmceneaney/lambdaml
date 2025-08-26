import hipopy.hipopy as hp
import os
import pandas as pd
import os
import numpy as np
import numpy.ma as ma
import awkward as ak
from tqdm import tqdm
import torch
import torch_geometric as tg
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url
import torch_geometric.transforms as T

#NOTE: NEW 2/20/23
from typing import List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

import modeloss

import itertools #NOTE: FOR DA TRAINING?  BUT IF YOU TRAIN DISCRIMINATOR FIRST YOU WON'T NEED THIS

#TODO: DEFINE TRANSFORMS
import torch_geometric.transforms as T

import matplotlib.pyplot as plt
# transform=T.KNNGraph(k=2) #NOTE: GRAPH.pos needs to be defined to use this
# transform=T.NormalizeFeatures()
# transform=T.Compose([T.KNNGraph(k=3,loop=False)])
# transform = T.Compose([T.KNNGraph(k=10,loop=True),NormalizeFeaturesNewTest(),T.ToUndirected(),T.AddSelfLoops()]) #T.AddSelfLoops(),
transform=None #NOTE: FOR SOME REASON T.ToUndirected() is changing the length of the graph labels??!?!?!?!?!?!RRRRRRG FRUSTRATING.
transform=T.NormalizeFeatures()

# Class definitions
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = None

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])

# Function definitions
def get_dataset(root='/work/clas12/users/mfmce/pyg_test_rec_particle_dataset_3_7_25/'):
    dataset = MyOwnDataset(
                root,
                transform=None, #T.Compose([T.ToUndirected(),T.KNNGraph(k=6)]),
                pre_transform=None,
                pre_filter=None
            )

    data = dataset[0]
    print('DEBUGGING: len(dataset)  = ',len(dataset))
    print('DEBUGGING: len(dataset.data)  = ',len(dataset.data))
    print('DEBUGGING: len(dataset.y)  = ',len(dataset.y))
    print('DEBUGGING: dataset.y.shape = ',dataset.y.shape)

    print(data.x.shape)
    print(data.x)
    # print(data.x[0])
    # print(data.x[0].sum())
    print(data.y.shape)
    print(data.y)
    print("DEBUGGING: data.x.dtype = ",data.x.dtype)
    print("DEBUGGING: data.y.dtype = ",data.y.dtype)
    print("DEBUGGING: data.kinematics.dtype = ",data.kinematics.dtype)
    print("DEBUGGING: data.y.shape = ",data.y.shape)
    print("DEBUGGING: data.y = ",data.y)
    g = tg.utils.to_networkx(data,to_undirected=True)
    import networkx as nx
    node_labels = {i:int(val.item()*100) for i, val in enumerate(data.x[:,0])}
    nx.draw(g,labels=node_labels)
    print("Going through dataset")
    for idx, d in enumerate(dataset):
        if d.y.shape[0]>2:
            print("d.y = ",d.y)
            print("idx = ",idx)
        if torch.any(torch.isnan(d.x)):
            print("DEBUGGING: nan @ idx, d = ",idx,d)
        if torch.any(torch.isinf(d.x)):
            print("DEBUGGING: nan @ idx, d = ",idx,d)
            
    print("DONE")

    l_sig = []
    l_bg = []
    print("DEBUGGING: dataset[0].y[0].item()== 0 = ",dataset[0].y[0].item()==0)
    for data in dataset:
    #     if len(data.y)!=2: print("DEBUGGING: len(data.y) = ",len(data.y))
        if data.y[0].item()==1:
            l_sig.append(data)
        else:
            l_bg.append(data)
    b_sig = torch_geometric.data.Batch().from_data_list(l_sig)
    b_bg = torch_geometric.data.Batch().from_data_list(l_bg)
    print(b_sig.x.shape)
    print(b_sig.y.shape)
    print(b_bg.x.shape)
    print(b_bg.y.shape)

    def plot_data_separated(array_sig,array_bg,title=None,xlabel='index',nbins=100,low=-1.1,high=1.1):
        
        array_sig = array_sig.flatten()
        array_bg = array_bg.flatten()

        # Plot MC-Matched distributions
        f = plt.figure()
        if title != None:
            plt.title(title)
        plt.title('MC Distribution')
        plt.hist(array_sig, color='r', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='signal')
        plt.hist(array_bg, color='b', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='background')
        plt.legend(loc='upper left', frameon=False)
        plt.ylabel('Counts')
        plt.xlabel(xlabel)
    #     f.savefig(xlabel+'_separated_'+todays_date+'.png')
        plt.show()
        f.savefig(xlabel+os.path.basename(root)+'.pdf')
        
    arr1 = b_sig.kinematics
    arr2 = b_bg.kinematics

    print("DEBUGGING: b_sig.y.shape = ",b_sig.y.shape)
    print("DEBUGGING; b_bg.y.shape = ",b_bg.y.shape)
        
    # arr1 = []
    # for el in b_sig.kinematics:
    #     arr1.extend(el)
    # arr2 = []
    # for el in b_bg.kinematics:
    #     arr2.extend(el)

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    print("DEBUGGING: type(arr1) = ",type(arr1))
    print("DEBUGGING: arr1.shape = ",arr1.shape)
    print("DEBUGGING: arr2.shape = ",arr2.shape)

    print("DEBUGGING: b_sig.x.dtype = ",b_sig.x.dtype)
    print("DEBUGGING: b_bg.x.dtype  = ",b_bg.x.dtype)

    print("DEBUGGING: b_sig.kinematics[0].dtype = ",b_sig.kinematics[0].dtype)
        
    # Plot data separated distributions
    plot_data_separated(arr1[:,3],arr2[:,3],xlabel="Q2",low=0.0,high=10.0)
    plot_data_separated(arr1[:,5],arr2[:,5],xlabel="W",low=0.0,high=5.0)
    plot_data_separated(arr1[:,6],arr2[:,6],xlabel="x",low=0.0,high=1.0)
    plot_data_separated(arr1[:,7],arr2[:,7],xlabel="y",low=0.0,high=1.0)
    plot_data_separated(arr1[:,8],arr2[:,8],xlabel="z",low=0.0,high=1.0)
    plot_data_separated(arr1[:,9],arr2[:,9],xlabel="xF",low=0.0,high=1.0)
    plot_data_separated(arr1[:,10],arr2[:,10],xlabel="mass",low=1.08,high=1.24)
    # plot_data_separated(arr1[:,4],arr2[:,4],xlabel="chi2")
    # plot_data_separated(arr1[:,5],arr2[:,5],xlabel="pid")
    # plot_data_separated(arr1[:,6],arr2[:,6],xlabel="status")

    return dataset

def get_dataset_dt(root='/work/clas12/users/mfmce/pyg_DATA_rec_particle_dataset_3_7_25/'):
    dataset = MyOwnDataset(
                root,
                transform=None, #T.Compose([T.ToUndirected(),T.KNNGraph(k=6)]),
                pre_transform=None,
                pre_filter=None
            )
    # DEBUGGING=True
    # if DEBUGGING:
    #     from torch_geometric.datasets import TUDataset
    #     dataset = TUDataset(root='/home/mfmce/drop', name='MUTAG')
    # print("DEBUGGING: len(dataset) = ",len(dataset))
    data = dataset[0]
    print('DEBUGGING: len(dataset)  = ',len(dataset))
    print('DEBUGGING: len(dataset.data)  = ',len(dataset.data))
    print('DEBUGGING: len(dataset.isdata)  = ',len(dataset.isdata))
    print('DEBUGGING: dataset.isdata.shape = ',dataset.isdata.shape)
    # print(data.x)
    # print(dataset)
    # if transform is not None: print(transform(dataset[0]).edge_index)
    # if transform is not None: print(transform(dataset[0]).x)
    print(data.x.shape)
    print(data.x)
    # print(data.x[0])
    # print(data.x[0].sum())
    print(data.isdata.shape)
    print(data.isdata)
    print("DEBUGGING: data.x.dtype = ",data.x.dtype)
    print("DEBUGGING: data.isdata.dtype = ",data.isdata.dtype)
    print("DEBUGGING: data.kinematics.dtype = ",data.kinematics.dtype)
    print("DEBUGGING: data.isdata.shape = ",data.isdata.shape)
    print("DEBUGGING: data.isdata = ",data.isdata)
    g = tg.utils.to_networkx(data,to_undirected=True)
    import networkx as nx
    node_labels = {i:int(val.item()*100) for i, val in enumerate(data.x[:,0])}
    nx.draw(g,labels=node_labels)
    print("Going through dataset")
    for idx, d in enumerate(dataset):
        if d.isdata.shape[0]>2:
            print("d.isdata = ",d.isdata)
            print("idx = ",idx)
        if torch.any(torch.isnan(d.x)):
            print("DEBUGGING: nan @ idx, d = ",idx,d)
        if torch.any(torch.isinf(d.x)):
            print("DEBUGGING: nan @ idx, d = ",idx,d)
            
    print("DONE")

    l_sig = []
    # l_bg = []
    print("DEBUGGING: dataset[0].isdata[0].item()== 0 = ",dataset[0].isdata[0].item()==0)
    for data in dataset:
    #     if len(data.isdata)!=2: print("DEBUGGING: len(data.isdata) = ",len(data.isdata))
        if data.isdata[0].item()==1:
            l_sig.append(data)
        else:
            l_bg.append(data)
    b_sig = torch_geometric.data.Batch().from_data_list(l_sig)
    # b_bg = torch_geometric.data.Batch().from_data_list(l_bg)
    print(b_sig.x.shape)
    print(b_sig.isdata.shape)
    # print(b_bg.x.shape)
    # print(b_bg.isdata.shape)

    def plot_data_separated(array_sig,title=None,xlabel='index',nbins=100,low=-1.1,high=1.1):
        
        array_sig = array_sig.flatten()

        # Plot MC-Matched distributions
        f = plt.figure()
        if title != None:
            plt.title(title)
        plt.title('Data Distribution')
        plt.hist(array_sig, color='r', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='Data')
        # plt.hist(array_bg, color='b', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='background')
        plt.legend(loc='upper left', frameon=False)
        plt.ylabel('Counts')
        plt.xlabel(xlabel)
    #     f.savefig(xlabel+'_separated_'+todays_date+'.png')
        plt.show()
        f.savefig(xlabel+os.path.basename(root)+'.pdf')
        
    arr1 = b_sig.kinematics
    # arr2 = b_bg.kinematics

    print("DEBUGGING: b_sig.isdata.shape = ",b_sig.isdata.shape)
    # print("DEBUGGING; b_bg.isdata.shape = ",b_bg.isdata.shape)
        
    # arr1 = []
    # for el in b_sig.kinematics:
    #     arr1.extend(el)
    # arr2 = []
    # for el in b_bg.kinematics:
    #     arr2.extend(el)

    arr1 = np.array(arr1)
    # arr2 = np.array(arr2)

    print("DEBUGGING: type(arr1) = ",type(arr1))
    print("DEBUGGING: arr1.shape = ",arr1.shape)
    # print("DEBUGGING: arr2.shape = ",arr2.shape)

    print("DEBUGGING: b_sig.x.dtype = ",b_sig.x.dtype)
    # print("DEBUGGING: b_bg.x.dtype  = ",b_bg.x.dtype)

    print("DEBUGGING: b_sig.kinematics[0].dtype = ",b_sig.kinematics[0].dtype)
        
    # Plot data separated distributions
    plot_data_separated(arr1[:,3],xlabel="Q2",low=0.0,high=10.0)
    plot_data_separated(arr1[:,5],xlabel="W",low=0.0,high=5.0)
    plot_data_separated(arr1[:,6],xlabel="x",low=0.0,high=1.0)
    plot_data_separated(arr1[:,7],xlabel="y",low=0.0,high=1.0)
    plot_data_separated(arr1[:,8],xlabel="z",low=0.0,high=1.0)
    plot_data_separated(arr1[:,9],xlabel="xF",low=0.0,high=1.0)
    plot_data_separated(arr1[:,10],xlabel="mass",low=1.08,high=1.24)
    # plot_data_separated(arr1[:,4],arr2[:,4],xlabel="chi2")
    # plot_data_separated(arr1[:,5],arr2[:,5],xlabel="pid")
    # plot_data_separated(arr1[:,6],arr2[:,6],xlabel="status")

    return dataset

dataset_mc = get_dataset(root='/work/clas12/users/mfmce/pyg_test_rec_particle_dataset_3_7_25/')
dataset_dt = get_dataset_dt(root='/work/clas12/users/mfmce/pyg_DATA_rec_particle_dataset_3_7_24/')
#dataset_both = TODO: SEE BELOW

# Model classes

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.norm import GraphNorm, BatchNorm

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
#         torch.manual_seed(12345)
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
#         print("x = ",x)
#         print("DEBUGGING: in GCN: begin: x.requires_grad = ",x.requires_grad)
        x = self.conv1(x, edge_index)
#         print("DEBUGGING: in GCN: self.conv1(x, edge_index): x.requires_grad = ",x.requires_grad)
        x = self.bn1(x)
#         print("DEBUGGING: in GCN: self.bn2(x): x.requires_grad = ",x.requires_grad)
#         print("self.conv1(x,edge_index) = ",x)
        x = x.relu()
#         x = torch.nn.function.elu(x)
#         print("DEBUGGING: in GCN: x.relu(): x.requires_grad = ",x.requires_grad)
#         print("x.relu() = ",x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
#         print("DEBUGGING: in GCN: self.bn2(x): x.requires_grad = ",x.requires_grad)
#         print("self.conv2(x,edge_index) = ",x)
        x = x.relu()
#         print("x.relu() = ",x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
#         print("self.conv3(x,edge_index) = ",x)
#         print("DEBUGGING: in GCN: self.bn3(x): x.requires_grad = ",x.requires_grad)

#         # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#         print("self.conv2(global_mean_pool(x, batch)) = ",x)
#         print("DEBUGGING: in GCN: global_mean_pool(x): x.requires_grad = ",x.requires_grad)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
#         print("DEBUGGING: in GCN: F.dropout: x.requires_grad = ",x.requires_grad)
#         print("F.dropout(x, p=0.5, training=self.training) = ",x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
#         print("DEBUGGING: in GCN: self.lin*(x): x.requires_grad = ",x.requires_grad)
#         print("self.lin3(x) = ",x)
#         x = torch.sigmoid(x) #NOTE: DON'T SOFTMAX IF USING BCELOSS, USE SIGMOID INSTEAD
#         print("torch.sigmoid(x) = ",x)
#         print("DEBUGGING: in GCN: torch.sigmoid(x): x.requires_grad = ",x.requires_grad)
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
    def __init__(self, in_channels=7, hidden_dim=64, num_layers=4, dropout=0.5, out_channels=2):

        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        for _ in range(self.num_layers - 1):
            mlp = MLP([in_channels if _==0 else self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim],norm='batch_norm',act='relu') #NOTE ADDED EXTRA LAYER HERE
            self.convs.append(GINConv(mlp, train_eps=False))
            self.bns.append(BatchNorm1d(self.hidden_dim))
#             in_channels = self.hidden_dim

        self.mlps = torch.nn.ModuleList()
        for _ in range(self.num_layers): #NOTE: INPUT DIM FOR MLP LAYER HAS TO MATCH INPUT DIM FOR EACH GRAPH REPRESENTATION IN EACH LAYER HERE
            self.mlps.append(MLP([in_channels if _==0 else self.hidden_dim, self.hidden_dim, self.hidden_dim, out_channels], norm=None, dropout=self.dropout, act='relu')) #NOTE ADDED EXTRA LAYER HERE AND..-> CHANGED ACTIVATION FUNCTION FROM DEFAULT RELU

    def forward(self, data):#x, edge_index, batch): #data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hidden_rep = [x]
        for i in range(self.num_layers - 1):
#             print("x = ",x.max().item(),x.min().item(),x.mean().item(),x.std().item())
            x = self.convs[i](x, edge_index)
#             print("x = conv(x, edge_index) = ",x.max().item(),x.min().item(),x.mean().item(),x.std().item())
#             print("i = ",i)
            x = self.bns[i](x)
#             print("x = bn(x) = ",x.max().item(),x.min().item(),x.mean().item(),x.std().item())
#             x = x.elu() #x.tanh() #x.relu()
            x = torch.nn.functional.relu(x)
#             print("x = x.relu() = ",x.max().item(),x.min().item(),x.mean().item(),x.std().item())
            hidden_rep.append(x)
#             raise RunTimeError
        score_over_layer = 0
        for i, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h,batch)#NOTE: ORIGINAL
#             pooled_h = global_mean_pool(h,batch)
            # pooled_h = global_max_pool(h,batch)
#             print("i, pooled_h = ",i," , ",pooled_h.max().item(),pooled_h.min().item(),pooled_h.mean().item(),pooled_h.std().item())
            newval = self.mlps[i](pooled_h)
            score_over_layer += newval#self.mlps[i](pooled_h)
#             print("self.mlps[i](pooled_h) = ",newval)
#             print("softmax(...) = ",torch.nn.functional.softmax(newval,dim=-1))
#         raise TypeError
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

# Create a domain adversarial model starting with the GIN base
hidden_dim = 64
out_channels = 2
model = GIN(in_channels=dataset_mc.num_node_features,hidden_dim=hidden_dim,num_layers=4,dropout=0.0,out_channels=hidden_dim)

# And then add a discriminator
nlayers_discriminator = 3
dropout = 0.0
discriminator = MLP([hidden_dim if idx<nlayers_discriminator-1 else out_channels for idx in range(nlayers_discriminator)], norm=None, dropout=dropout, act='relu')

# And then a classifier
nlayers_classifier = 3
dropout = 0.0
classifier = MLP([hidden_dim if idx<nlayers_classifier-1 else out_channels for idx in range(nlayers_classifier)], norm=None, dropout=dropout, act='relu')

#TODO: PUT MODELS ON DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device = ",device)
model = model.to(device)
discriminator = discriminator.to(device)
classifier = classifier.to(device)
print("DEBUGGING: torch.cuda.is_available() = ",torch.cuda.is_available())

#TODO: SPLIT DATASET
from torch.utils.data import random_split, ConcatDataset #TODO: SEE IF YOU CAN USE THIS

dataset_both = ConcatDataset([dataset_mc,dataset_dt])

def get_loaders(dataset):
    # torch.manual_seed(12345)
    print('DEBUGGING: BEFORE: dataset.y.shape = ',dataset.y.shape)
    dataset = dataset.shuffle()
    print('DEBUGGING: AFTER:  dataset.y.shape = ',dataset.y.shape)

    print(len(dataset))

    fracs = [0.8, 0.1, 0.1] #NOTE: SHOULD CHECK np.sum(fracs) == 1 and len(fracs)==3
    fracs = [torch.sum(torch.tensor(fracs[:idx])) for idx in range(1,len(fracs)+1)]
    print(fracs)
    split1, split2 = [int(len(dataset)*frac) for frac in fracs[:-1]]
    train_dataset = dataset[:split1]
    val_dataset = dataset[split1:split2]
    test_dataset = dataset[split2:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(val_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    print("train_dataset")
    for idx, d in enumerate(train_dataset):
        if d.y.shape[0]>2:
            print("d.y = ",d.y)
            print("idx = ",idx)
            
    print("val_dataset")
    for idx, d in enumerate(val_dataset):
        if d.y.shape[0]>2:
            print("d.y = ",d.y)
            print("idx = ",idx)
            
    print("test_dataset")
    for idx, d in enumerate(test_dataset):
        if d.y.shape[0]>2:
            print("d.y = ",d.y)
            print("idx = ",idx)

    #TODO: CREATE DATALOADERS
    from torch_geometric.loader import DataLoader
    from torch.utils.data import WeightedRandomSampler
    print("DEBUGGING: train_dataset.y.shape = ",train_dataset.y.shape)
    _, train_counts = np.unique(train_dataset.y, return_counts=True)
    print("DEBUGGING: np.unique(train_dataset) = ",_,train_counts)
    train_weights = [1/train_counts[i] for i in train_dataset.y]
    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_dataset), replacement=True)
    _, val_counts = np.unique(val_dataset.y, return_counts=True)
    print("DEBUGGING: np.unique(val_dataset) = ",_,val_counts)
    val_weights = [1/val_counts[i] for i in val_dataset.y]
    val_sampler = WeightedRandomSampler(weights=val_weights, num_samples=len(val_dataset), replacement=True)
    _, test_counts = np.unique(test_dataset.y, return_counts=True)
    print("DEBUGGING: np.unique(test_dataset) = ",_,test_counts)
    test_weights = [1/test_counts[i] for i in test_dataset.y]
    test_sampler = WeightedRandomSampler(weights=test_weights, num_samples=len(test_dataset), replacement=True)

    batch_size = 256
    use_weighted_samplers = False
    if not use_weighted_samplers:
        train_sampler, val_sampler, test_sampler = None, None, None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=not use_weighted_samplers)#, drop_last=True)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)#NOTE: #TODO: Try no sampling here for evaluation...

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()
        break

    # print("train_loader")
    # for idx, d in enumerate(train_loader):
    #     if d.y.shape[0]!=batch_size:
    #         print("d.y = ",d.y)
    #         print("idx = ",idx)
            
    # print("val_loader")
    # for idx, d in enumerate(val_loader):
    #     if d.y.shape[0]!=batch_size:
    #         print("d.y = ",d.y)
    #         print("idx = ",idx)
            
    # print("test_loader")
    # for idx, d in enumerate(test_loader):
    #     if d.y.shape[0]!=batch_size:
    #         print("d.y = ",d.y)
    #         print("idx = ",idx)

    return (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset)

def get_loaders_dt_mc(ds_mc, ds_dt):
    # torch.manual_seed(12345)
    # print('DEBUGGING: BEFORE: dataset.isdata.shape = ',dataset.isdata.shape)
    ds_mc = ds_mc.shuffle()
    ds_dt = ds_dt.shuffle()
    # print('DEBUGGING: AFTER:  dataset.isdata.shape = ',dataset.isdata.shape)

    print(len(ds_mc))
    print(len(ds_dt))

    fracs = [0.8, 0.1, 0.1] #NOTE: SHOULD CHECK np.sum(fracs) == 1 and len(fracs)==3
    fracs = [torch.sum(torch.tensor(fracs[:idx])) for idx in range(1,len(fracs)+1)]
    print(fracs)
    ds_size = len(ds_mc) + len(ds_dt)
    print("ds_size (full) = ",ds_size)
    split1, split2 = [int(len(ds_mc)*frac) for frac in fracs[:-1]]
    train_ds_mc = ds_mc[:split1]
    val_ds_mc = ds_mc[split1:split2]
    test_ds_mc = ds_mc[split2:]

    split1, split2 = [int(len(ds_dt)*frac) for frac in fracs[:-1]]
    train_ds_dt = ds_dt[:split1]
    val_ds_dt = ds_dt[split1:split2]
    test_ds_dt = ds_dt[split2:]

    train_dataset = ConcatDataset([train_ds_mc,train_ds_dt])
    val_dataset   = ConcatDataset([val_ds_mc,val_ds_dt])
    test_dataset  = ConcatDataset([test_ds_mc,test_ds_dt])

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(val_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    # #TODO: CREATE DATALOADERS
    # from torch_geometric.loader import DataLoader
    # from torch.utils.data import WeightedRandomSampler
    # _, train_counts = np.unique([train_dataset_el.isdata for train_dataset_el in train_dataset], return_counts=True)
    # print("DEBUGGING: np.unique(train_dataset) = ",_,train_counts)
    # train_weights = [1/train_counts[i] for i in train_dataset]
    # train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_dataset), replacement=True)
    # _, val_counts = np.unique([val_dataset_el.isdata for val_dataset_el in val_dataset], return_counts=True)
    # print("DEBUGGING: np.unique(val_dataset) = ",_,val_counts)
    # val_weights = [1/val_counts[i] for i in val_dataset]
    # val_sampler = WeightedRandomSampler(weights=val_weights, num_samples=len(val_dataset), replacement=True)
    # _, test_counts = np.unique([test_dataset_el.isdata for test_dataset_el in test_dataset], return_counts=True)
    # print("DEBUGGING: np.unique(test_dataset) = ",_,test_counts)
    # test_weights = [1/test_counts[i] for i in test_dataset]
    # test_sampler = WeightedRandomSampler(weights=test_weights, num_samples=len(test_dataset), replacement=True)
    train_sampler, val_sampler, test_sampler = None, None, None
    
    batch_size = 256
    use_weighted_samplers = False
    if not use_weighted_samplers:
        train_sampler, val_sampler, test_sampler = None, None, None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=not use_weighted_samplers)#, drop_last=True)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)#NOTE: #TODO: Try no sampling here for evaluation...

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()
        break

    # print("train_loader")
    # for idx, d in enumerate(train_loader):
    #     if d.isdata.shape[0]!=batch_size:
    #         print("d.isdata = ",d.isdata)
    #         print("idx = ",idx)
            
    # print("val_loader")
    # for idx, d in enumerate(val_loader):
    #     if d.isdata.shape[0]!=batch_size:
    #         print("d.isdata = ",d.isdata)
    #         print("idx = ",idx)
            
    # print("test_loader")
    # for idx, d in enumerate(test_loader):
    #     if d.isdata.shape[0]!=batch_size:
    #         print("d.isdata = ",d.isdata)
    #         print("idx = ",idx)

    return (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset)

# Get data loaders
loaders_mc, datasets_mc = get_loaders(dataset_mc)
loaders_both, datasets_both = get_loaders_dt_mc(dataset_mc, dataset_dt)
batch_size = 256
use_weighted_samplers = False
train_dataset_dt = datasets_both[0]
train_dataset_mc = datasets_mc[0]

print(len(dataset_mc.isdata))
print(len(dataset_mc.y))
print(len(dataset_dt.isdata))
print(len(dataset_dt.y))
print(dataset_both[0].y)


#TODO: DO THIS FOR BOTH Y AND ISDATA ATTRIBUTES
# # Instantiate model, optimizer, and loss function
# model = GCN(dataset.num_node_features,64,dataset.num_classes).to(device)
# Compile model
# model = torch_geometric.compile(model, dynamic=True)#NOTE: Not sure why this throws error now...

# Create optimizer for GIN and DISCR
# print("DEBUGGING: list(model.parameters()         = ",list(model.parameters()))
# print("DEBUGGING: list(discriminator.parameters() = ",list(discriminator.parameters()))
# print("DEBUGGING: list(model.parameters(),discriminator.parameters()) = ",list(model.parameters(),discriminator.parameters()))
optimizer_mo = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer_di = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
optimizer_cl = torch.optim.Adam(classifier.parameters(), lr=1e-3)
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# print("DEBUGGING: y[0] = ",torch_geometric.data.Batch().from_data_list(train_dataset_dt).isdata[0])
# print("DEBUGGING: len(train_dataset_dt) = ",len(train_dataset_dt))
# print("DEBUGGING: len(batch(train_dataset_dt)) = ",len(torch_geometric.data.Batch().from_data_list(train_dataset_dt)))
# print("DEBUGGING: y.shape = ",torch_geometric.data.Batch().from_data_list(train_dataset_dt).isdata.shape)
# print("DEBUGGING: y[0:10] = ",torch_geometric.data.Batch().from_data_list(train_dataset_dt).isdata[0:10])
# # data_labels = torch_geometric.data.Batch().from_data_list(train_dataset).isdata[:,0] #NOTE: THIS IS FOR 2D Labels
unique_dt, counts_dt = np.unique([el.isdata for el in train_dataset_dt],return_counts=True)
print("DEBUGGING: unique_dt, counts_dt = ",unique_dt,counts_dt)
weight_signal_dt = counts_dt[1]/counts_dt[0]#DEBUGGING MULTIPLY BY 2 ...
print("weight_signal_dt = ",weight_signal_dt)
weight_dt = torch.FloatTensor([1.0, 1.0/weight_signal_dt]).to(device) #NOTE: That labels are [sg?,bg?] so label 0 in this case is sg and label 1 is bg.
print("DEBUGGING: weigh_dt = ",weight_dt)
criterion_dt = torch.nn.CrossEntropyLoss(weight=weight_dt if not use_weighted_samplers else None,reduction='mean')

# Create optimizer for GIN AND CLASSIFIER
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# print("DEBUGGING: y[0] = ",torch_geometric.data.Batch().from_data_list(train_dataset_mc).y[0])
# print("DEBUGGING: len(train_dataset_mc) = ",len(train_dataset_mc))
# print("DEBUGGING: len(batch(train_dataset_mc)) = ",len(torch_geometric.data.Batch().from_data_list(train_dataset_mc)))
# print("DEBUGGING: y.shape = ",torch_geometric.data.Batch().from_data_list(train_dataset_mc).y.shape)
# print("DEBUGGING: y[0:10] = ",torch_geometric.data.Batch().from_data_list(train_dataset_mc).y[0:10])
# # data_labels = torch_geometric.data.Batch().from_data_list(train_dataset).y[:,0] #NOTE: THIS IS FOR 2D Labels
unique_mc, counts_mc = np.unique([el.y for el in train_dataset_mc],return_counts=True)
print("DEBUGGING: unique_mc, counts_mc = ",unique_mc,counts_mc)
weight_signal_mc = counts_mc[1]/counts_mc[0]#DEBUGGING MULTIPLY BY 2 ...
print("weight_signal_mc = ",weight_signal_mc)
weight_mc = torch.FloatTensor([weight_signal_mc, 1.0]).to(device) #NOTE: That labels are [sg?,bg?] so label 0 in this case is sg and label 1 is bg.
print("DEBUGGING: weight_mc = ",weight_mc)
criterion_mc = torch.nn.CrossEntropyLoss(weight=weight_mc if not use_weighted_samplers else None,reduction='mean')

model_dt = torch.nn.Sequential(
    model,
    discriminator
)

model_mc = torch.nn.Sequential(
    model,
    classifier
)

# Define training and testing routines
def train(model,optimizers,loader,criterion):
    model.train()

    for idx, data in tqdm(enumerate(loader)):  # Iterate in batches over the training dataset.
        data = data.to(device)
        for optimizer in optimizers:
            optimizer.zero_grad()  # Clear gradients.
#         x = data.x
#         y = data.y
#         if batch_size!=y.shape[0]:
#             print("DEBUGGING: x.shape = ",x.shape)
#             print("DEBUGGING: len(x) = ",len(x))
#             print("DEBUGGING: y.shape = ",y.shape)
#             print("DEBUGGING: len(y) = ",len(y))
# #             print("DEBUGGING: x = ",x)
#             print("DEBUGGING: y = ",y)
#             print("DEBUGGING: idx = ",idx)
# #             print("DEBUGGING: edge_index = ",data.edge_index)
# #         edge_index = data.edge_index
# #         batch = data.batch
        out = model(data)#data.x, data.edge_index, data.batch)  # Perform a single forward pass.
#         out = torch.nn.functional.softmax(out,dim=-1)
#         print("DEBUGGING: out.dtype = ",out.dtype)
#         print("DEBUGGING: data.y.dtype = ",data.y.dtype)
#         print("DEBUGGING: out.shape = ",out.shape)
#         print("DEBUGGING: data.y.shape = ",data.y.shape)
#         print("DEBUGGING: out = ",out)
#         print("DEBUGGING: y   = ",data.y)
#         print("DEBUGGING: y.shape   = ",data.y.shape)
#         break
#         print("DEBUGGING: out.device = ",out.device)
#         print("DEBUGGING: data.y.device = ",data.y.device)
#         print("DEBUGGING: device = ",device)
#         print("DEBUGGING: out.shape = ",out.shape)
#         print("DEBUGGING: data.y.shape = ",data.y.shape)
#         print("DEBUGGING: out.dtype = ",out.dtype)
#         print("DEBUGGING: data.y.dtype = ",data.y.dtype)
#         print("DEBUGGING: data.kinematics = ",data.kinematics)
        #if idx<5: print("DEBUGGING: data.kinematics.shape = ",data.kinematics.shape)
#         print("DEBUGGING: tg.utils.unbatch(data.src,data.)")
        """masses = data.kinematics[:,10]
        my_mean = 1.08
        my_sig = (1.24-1.08)/2
        masses -= (my_mean+my_sig)
        masses /= my_sig
        mloss_coeff = 1.0 #0.033"""
        #if idx<5: print("DEBUGGING: masses.shape = ",masses.shape)
#         loss = mloss_coeff * flatness_loss(out,data.y,masses)+criterion(out,data.y)
        loss = criterion(out, data.y)  # Compute the loss. #TODO: Add attributes to data to say whether it is data or MC
        #TODO: Try test traiining with sequential model and then freezing and adding classifier and propagating discriminator loss
        """if idx<5:
            print("loss = ",loss)
            fl = flatness_loss(out,data.y,masses)
            print("fl = ",fl)
        loss += mloss_coeff * flatness_loss(out,data.y,masses)
        if idx<5:
            print("updated loss = ",loss)"""
#         break#DEBUGGING!
#         out = torch.nn.functional.softmax(out,dim=-1)
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
        loss.backward()  # Derive gradients.
        for optimizer in optimizers:
            optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.

# Define training and testing routines
def train_discr(model,optimizers,loader,criterion):
    model.train()

    for idx, data in tqdm(enumerate(loader)):  # Iterate in batches over the training dataset.
        data = data.to(device)
        for optimizer in optimizers:
            optimizer.zero_grad()  # Clear gradients.
#         x = data.x
#         y = data.isdata
#         if batch_size!=y.shape[0]:
#             print("DEBUGGING: x.shape = ",x.shape)
#             print("DEBUGGING: len(x) = ",len(x))
#             print("DEBUGGING: y.shape = ",y.shape)
#             print("DEBUGGING: len(y) = ",len(y))
# #             print("DEBUGGING: x = ",x)
#             print("DEBUGGING: y = ",y)
#             print("DEBUGGING: idx = ",idx)
# #             print("DEBUGGING: edge_index = ",data.edge_index)
# #         edge_index = data.edge_index
# #         batch = data.batch
        out = model(data)#data.x, data.edge_index, data.batch)  # Perform a single forward pass.
#         out = torch.nn.functional.softmax(out,dim=-1)
#         print("DEBUGGING: out.dtype = ",out.dtype)
#         print("DEBUGGING: data.isdata.dtype = ",data.isdata.dtype)
#         print("DEBUGGING: out.shape = ",out.shape)
#         print("DEBUGGING: data.isdata.shape = ",data.isdata.shape)
#         print("DEBUGGING: out = ",out)
#         print("DEBUGGING: y   = ",data.isdata)
#         print("DEBUGGING: y.shape   = ",data.isdata.shape)
#         break
#         print("DEBUGGING: out.device = ",out.device)
#         print("DEBUGGING: data.isdata.device = ",data.isdata.device)
#         print("DEBUGGING: device = ",device)
#         print("DEBUGGING: out.shape = ",out.shape)
#         print("DEBUGGING: data.isdata.shape = ",data.isdata.shape)
#         print("DEBUGGING: out.dtype = ",out.dtype)
#         print("DEBUGGING: data.isdata.dtype = ",data.isdata.dtype)
#         print("DEBUGGING: data.kinematics = ",data.kinematics)
        #if idx<5: print("DEBUGGING: data.kinematics.shape = ",data.kinematics.shape)
#         print("DEBUGGING: tg.utils.unbatch(data.src,data.)")
        """masses = data.kinematics[:,10]
        my_mean = 1.08
        my_sig = (1.24-1.08)/2
        masses -= (my_mean+my_sig)
        masses /= my_sig
        mloss_coeff = 1.0 #0.033"""
        #if idx<5: print("DEBUGGING: masses.shape = ",masses.shape)
#         loss = mloss_coeff * flatness_loss(out,data.isdata,masses)+criterion(out,data.isdata)
        loss = criterion(out, data.isdata)  # Compute the loss. #TODO: Add attributes to data to say whether it is data or MC
        #TODO: Try test traiining with sequential model and then freezing and adding classifier and propagating discriminator loss
        """if idx<5:
            print("loss = ",loss)
            fl = flatness_loss(out,data.isdata,masses)
            print("fl = ",fl)
        loss += mloss_coeff * flatness_loss(out,data.isdata,masses)
        if idx<5:
            print("updated loss = ",loss)"""
#         break#DEBUGGING!
#         out = torch.nn.functional.softmax(out,dim=-1)
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
        loss.backward()  # Derive gradients.
        for optimizer in optimizers:
            optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.


# Define training and testing routines
def train_da(model,classifier,optimizers,train_loader,criterion,discriminator,optimizer_discr,criterion_discr,alpha):
    model.train()
    classifier.train()
    discriminator.train() #NOTE: NEEDS TO BE IN TRAIN MODE TO COMPUTE GRADIENTS

    for idx, data in tqdm(enumerate(train_loader)):  # Iterate in batches over the training dataset.
        data = data.to(device)
        for optimizer in optimizers:
            optimizer.zero_grad()  # Clear gradients.
        optimizer_discr.zero_grad()  # Clear gradients.
#         x = data.x
#         y = data.y
#         if batch_size!=y.shape[0]:
#             print("DEBUGGING: x.shape = ",x.shape)
#             print("DEBUGGING: len(x) = ",len(x))
#             print("DEBUGGING: y.shape = ",y.shape)
#             print("DEBUGGING: len(y) = ",len(y))
# #             print("DEBUGGING: x = ",x)
#             print("DEBUGGING: y = ",y)
#             print("DEBUGGING: idx = ",idx)
# #             print("DEBUGGING: edge_index = ",data.edge_index)
# #         edge_index = data.edge_index
# #         batch = data.batch
        out_model = model(data) #data.x, data.edge_index, data.batch) # Perform a single forward pass.
        out = classifier(out_model)
        out_discr = discriminator(out_model)
#         out = torch.nn.functional.softmax(out,dim=-1)
#         print("DEBUGGING: out.dtype = ",out.dtype)
#         print("DEBUGGING: data.y.dtype = ",data.y.dtype)
#         print("DEBUGGING: out.shape = ",out.shape)
#         print("DEBUGGING: data.y.shape = ",data.y.shape)
#         print("DEBUGGING: out = ",out)
#         print("DEBUGGING: y   = ",data.y)
#         print("DEBUGGING: y.shape   = ",data.y.shape)
#         break
#         print("DEBUGGING: out.device = ",out.device)
#         print("DEBUGGING: data.y.device = ",data.y.device)
#         print("DEBUGGING: device = ",device)
#         print("DEBUGGING: out.shape = ",out.shape)
#         print("DEBUGGING: data.y.shape = ",data.y.shape)
#         print("DEBUGGING: out.dtype = ",out.dtype)
#         print("DEBUGGING: data.y.dtype = ",data.y.dtype)
#         print("DEBUGGING: data.kinematics = ",data.kinematics)
        #if idx<5: print("DEBUGGING: data.kinematics.shape = ",data.kinematics.shape)
#         print("DEBUGGING: tg.utils.unbatch(data.src,data.)")
        """masses = data.kinematics[:,10]
        my_mean = 1.08
        my_sig = (1.24-1.08)/2
        masses -= (my_mean+my_sig)
        masses /= my_sig
        mloss_coeff = 1.0 #0.033"""
        #if idx<5: print("DEBUGGING: masses.shape = ",masses.shape)
#         loss = mloss_coeff * flatness_loss(out,data.y,masses)+criterion(out,data.y)
        loss = criterion(out, data.y)  # Compute the loss. #TODO: Add attributes to data to say whether it is data or MC
        loss_discr = criterion_discr(out, data.isdata)
        #TODO: Try test traiining with sequential model and then freezing and adding classifier and propagating discriminator loss
        """if idx<5:
            print("loss = ",loss)
            fl = flatness_loss(out,data.y,masses)
            print("fl = ",fl)
        loss += mloss_coeff * flatness_loss(out,data.y,masses)
        if idx<5:
            print("updated loss = ",loss)"""
#         break#DEBUGGING!
#         out = torch.nn.functional.softmax(out,dim=-1)
#         pred = out.argmax(dim=1)  # Use the class with highest probability.

        tot_loss = loss - alpha * loss_discr
        tot_loss.backward()  # Derive gradients.
        for optimizer in optimizers:
            optimizer.step()#NOTE: ONLY STEP THE MODEL AND CLASSIFIER OPTIMIZER  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.
        
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
@torch.no_grad()
def test(model,optimizer,loader,criterion):
    model.eval()
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             print("DEBUGGING: name, param = ",name,param)
    correct = 0
    loss_ = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    decisions = []
    outputs = []
    y_true = []
    for data in tqdm(loader):  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data) #data.x, data.edge_index, data.batch)
#         out = torch.nn.functional.softmax(out,dim=-1)
        loss = criterion(out, data.y)
        loss_ += loss.item()
        out = torch.nn.functional.softmax(out,dim=-1)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
#         print("DEBUGGING: np.unique(pred, return_counts=True) = ",np.unique(pred.cpu(),return_counts=True))
#         print("DEBUGGING: pred.shape = ",pred.shape)
#         print("DEBUGGING: data.y.shape = ",data.y.shape)
        correct += int((pred == data.y).sum())  # Check against ground-truth labels. #NOTE: THAT NEED index :,1 since labels are 0,1 (pred==1) if bg and 1,0 (pred==0) if true
#         tp += int(torch.logical_and(pred==data.y,pred==1).sum())#NOTE: THIS ONLY WORKS FOR BINARY CLASSIFICATION
#         fp += int(torch.logical_and(pred!=data.y,pred==1).sum())
#         fn += int(torch.logical_and(pred!=data.y,pred==0).sum())
#         tn += int(torch.logical_and(pred==data.y,pred==0).sum())
        cm = confusion_matrix(data.y.cpu(),pred.cpu(),labels=[0,1]) #NOTE: SAME INDEXING DOWN HEREE AS WHEN ASSIGNING CORRECT ABOVE.
        tp += cm[1,1]
        fp += cm[0,1] # bg classified as sig is 0,1
        fn += cm[1,0] # signal classified as bg is 1,0
        tn += cm[0,0]
        
        outputs.extend(out.cpu())
        decisions.extend(pred.cpu())
        y_true.extend(data.y.cpu())
        
    precision = tp / (tp + fp) # accuracy of the identified signal events
    recall = tp / (tp + fn) # efficiency
    precision_n = tn / (tn + fn)
    recall_n = tn / (tn + fp)
    
    roc_auc = roc_auc_score(y_true,decisions)
#     print("tp = ",tp)
#     print("fp = ",fp)
#     print("fn = ",fn)
#     print("tn = ",tn)
#     print("len(loader.dataset) = ",len(loader.dataset))
#     print("acc = ",correct/len(loader.dataset))
#     print("(tp+tn)/(tp+fp+tn+fn) = ",(tp+tn)/(tp+fp+tn+fn))
    return correct / len(loader.dataset), loss_ / len(loader.dataset), precision, recall, precision_n, recall_n, outputs, decisions, y_true, roc_auc # Derive ratio of correct predictions.

# acc, loss = test(train_loader)
# print(acc,loss)
# acc, loss = test(val_loader)
# print(acc, loss)

@torch.no_grad()
def test_discr(model,optimizer,loader,criterion):
    model.eval()
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             print("DEBUGGING: name, param = ",name,param)
    correct = 0
    loss_ = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    decisions = []
    outputs = []
    y_true = []
    for data in tqdm(loader):  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data) #data.x, data.edge_index, data.batch)
#         out = torch.nn.functional.softmax(out,dim=-1)
        loss = criterion(out, data.isdata)
        loss_ += loss.item()
        out = torch.nn.functional.softmax(out,dim=-1)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
#         print("DEBUGGING: np.unique(pred, return_counts=True) = ",np.unique(pred.cpu(),return_counts=True))
#         print("DEBUGGING: pred.shape = ",pred.shape)
#         print("DEBUGGING: data.isdata.shape = ",data.isdata.shape)
        correct += int((pred == data.isdata).sum())  # Check against ground-truth labels. #NOTE: THAT NEED index :,1 since labels are 0,1 (pred==1) if bg and 1,0 (pred==0) if true
#         tp += int(torch.logical_and(pred==data.isdata,pred==1).sum())#NOTE: THIS ONLY WORKS FOR BINARY CLASSIFICATION
#         fp += int(torch.logical_and(pred!=data.isdata,pred==1).sum())
#         fn += int(torch.logical_and(pred!=data.isdata,pred==0).sum())
#         tn += int(torch.logical_and(pred==data.isdata,pred==0).sum())
        cm = confusion_matrix(data.isdata.cpu(),pred.cpu(),labels=[0,1]) #NOTE: SAME INDEXING DOWN HEREE AS WHEN ASSIGNING CORRECT ABOVE.
        tp += cm[1,1]
        fp += cm[0,1] # bg classified as sig is 0,1
        fn += cm[1,0] # signal classified as bg is 1,0
        tn += cm[0,0]
        
        outputs.extend(out.cpu())
        decisions.extend(pred.cpu())
        y_true.extend(data.isdata.cpu())
        
    precision = tp / (tp + fp) # accuracy of the identified signal events
    recall = tp / (tp + fn) # efficiency
    precision_n = tn / (tn + fn)
    recall_n = tn / (tn + fp)
    
    roc_auc = roc_auc_score(y_true,decisions)
#     print("tp = ",tp)
#     print("fp = ",fp)
#     print("fn = ",fn)
#     print("tn = ",tn)
#     print("len(loader.dataset) = ",len(loader.dataset))
#     print("acc = ",correct/len(loader.dataset))
#     print("(tp+tn)/(tp+fp+tn+fn) = ",(tp+tn)/(tp+fp+tn+fn))
    return correct / len(loader.dataset), loss_ / len(loader.dataset), precision, recall, precision_n, recall_n, outputs, decisions, y_true, roc_auc # Derive ratio of correct predictions.

# acc, loss = test(train_loader)
# print(acc,loss)
# acc, loss = test(val_loader)
# print(acc, loss)

train_loader_mc, val_loader_mc, test_loader_mc = loaders_mc
train_loader_dt, val_loader_dt, test_loader_dt = loaders_both

#---------- TODO: TRAIN THE MODEL+DISCRIMINATOR ----------#
outdir = '/work/clas12/users/mfmce/lambdaml/'

# Train model batch size 64 lr 0.001 epochs 30
nepochs = 30 #NOTE: FOR TESTING.  FOR DEPLOYMENT USE A LARGER NUMBER LIKE 30
train_metrics = {"acc":[], "loss":[], "precision":[], "recall":[], "roc_auc":[]}
val_metrics = {"acc":[], "loss":[], "precision":[], "recall":[], "roc_auc":[]}
# for name, param in model.named_parameters():
#     if 'weight' in name:
#         print("DEBUGGING: name, param = ",name,param)
#     break
model_best_auc = None
roc_aucs = []
for epoch in range(nepochs):
    print("BEFORE TRAIN()")
    train_discr(model_dt,[optimizer_di,optimizer_mo],train_loader_dt,criterion_dt)
    print("BEFORE TEST(TRAIN_LOADER)")
    train_acc, train_loss, train_precision, train_recall, train_precision_n, train_recall_n, _, _, __, train_roc_auc = test_discr(model_dt,[optimizer_di,optimizer_mo],train_loader_dt,criterion_dt)
    train_metrics["acc"].append(train_acc)
    train_metrics["loss"].append(train_loss)
    train_metrics["precision"].append(train_precision)
    train_metrics["recall"].append(train_recall)
    train_metrics["roc_auc"].append(train_roc_auc)
    print("BEFORE TEST(VAL_LOADER)")
    val_acc, val_loss, val_precision, val_recall, val_precision_n, val_recall_n, _, _, __, val_roc_auc = test_discr(model_dt,[optimizer_di,optimizer_mo],val_loader_dt,criterion_dt)
    if epoch==0 or val_roc_auc>np.max(val_metrics["roc_auc"]):
        PATH = os.path.join(outdir,'phase_1_model_best_auc.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_mo.state_dict(),
#             'loss': loss,
            }, PATH)
#         PATH = os.path.join(outdir,'classifier_best_auc.pt')
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': classifier.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
# #             'loss': loss,
#             }, PATH)
        PATH = os.path.join(outdir,'phase_1_discriminator_best_auc.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_di.state_dict(),
#             'loss': loss,
            }, PATH)
    val_metrics["acc"].append(val_acc)
    val_metrics["loss"].append(val_loss)
    val_metrics["precision"].append(val_precision)
    val_metrics["recall"].append(val_recall)
    val_metrics["roc_auc"].append(val_roc_auc)
    print("DISCRIMINATOR: Epoch ",epoch," Train acc: ",train_acc," loss: ",train_loss," precision: ",train_precision," recall: ",train_recall)#," precision_n: ",train_precision_n," recall_n: ",train_recall_n)
    print("DISCRIMINATOR: Epoch ",epoch," Val   acc: ",val_acc,  " loss: ",val_loss," precision: ",val_precision," recall: ",val_recall)#," precision_n: ",val_precision_n," recall_n: ",val_recall_n)

# for name, param in model.named_parameters():
#     if 'weight' in name:
#         print("DEBUGGING: name, param = ",name,param)
#     break
import matplotlib.pyplot as plt
epochs = [i for i in range(len(train_metrics["loss"]))]
f = plt.figure()
ymax = max(max(train_metrics['loss']),max(val_metrics['loss']))
ymax = 0.01 #max(max(train_metrics['loss']),max(val_metrics['loss']))
print("ymax = ",ymax)
plt.ylim((10**-3,10**-1))
plt.semilogy(epochs, train_metrics['loss'], color='tab:orange', linewidth=2, markersize=1, label="Training")
plt.semilogy(epochs, val_metrics['loss'], color='tab:blue', linewidth=2, markersize=1, label="Validation")
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Loss')
f.savefig(os.path.join(outdir,'loss_discr.pdf'))
plt.close()

f = plt.figure()
plt.ylim(1e-5,1)
plt.plot(epochs, train_metrics['acc'], color='tab:orange', linewidth=2, markersize=1, label="Training")
plt.plot(epochs, val_metrics['acc'], color='tab:blue', linewidth=2, markersize=1, label="Validation")
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
f.savefig(os.path.join(outdir,'acc_discr.pdf'))
plt.close()

f = plt.figure()
plt.ylim(0,1)
plt.plot(epochs, train_metrics['precision'], color='tab:orange', linewidth=2, markersize=1, label="Training")
plt.plot(epochs, val_metrics['precision'], color='tab:blue', linewidth=2, markersize=1, label="Validation")
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Precision')
f.savefig(os.path.join(outdir,'prec_discr.pdf'))
plt.close()

f = plt.figure()
plt.ylim(0,1)
plt.plot(epochs, train_metrics['recall'], color='tab:orange', linewidth=2, markersize=1, label="Training")
plt.plot(epochs, val_metrics['recall'], color='tab:blue', linewidth=2, markersize=1, label="Validation")
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Recall')
f.savefig(os.path.join(outdir,'recall_discr.pdf'))
plt.close()

# Save logs to yaml
import yaml

with open(os.path.join(outdir,'train_metrics_discr.yaml'), 'w') as file:
    yaml.dump(train_metrics, file)
with open(os.path.join(outdir,'val_metrics_discr.yaml'), 'w') as file:
    yaml.dump(val_metrics, file)

#NOTE: ALSO TRY CATBOOST WITH NEW VERTEXING
#NOTE: ALSO TRY GIN/PNet WITH NEW VERTEXING
#NOTE: SEE BELOW FIRST: CAN ALSO IMPLEMENT IN PL SO YOU CAN USE MULTIPLE GPUs
#NOTE: THIS WILL TAKE ROUGHLY 21min for 497000 training samples without GPU so just install GPU version of pytorch and save time.


#---------- TODO: TRAIN THE MODEL+DISCRIMINATOR ----------#
outdir = '/work/clas12/users/mfmce/lambdaml/'
alpha  = 0.1

# Train model batch size 64 lr 0.001 epochs 30
nepochs = 30 #NOTE: FOR TESTING.  FOR DEPLOYMENT USE A LARGER NUMBER LIKE 30
train_metrics = {"acc":[], "loss":[], "precision":[], "recall":[], "roc_auc":[]}
val_metrics = {"acc":[], "loss":[], "precision":[], "recall":[], "roc_auc":[]}
train_metrics_da = {"acc":[], "loss":[], "precision":[], "recall":[], "roc_auc":[]}
val_metrics_da = {"acc":[], "loss":[], "precision":[], "recall":[], "roc_auc":[]}
# for name, param in model.named_parameters():
#     if 'weight' in name:
#         print("DEBUGGING: name, param = ",name,param)
#     break
model_best_auc = None
roc_aucs = []
for epoch in range(nepochs):
    print("BEFORE TRAIN()")
    train_da(model,classifier,[optimizer_cl,optimizer_mo],train_loader_mc,criterion_mc,discriminator,optimizer_di,criterion_dt,alpha)
    print("BEFORE TEST(TRAIN_LOADER)")
    train_acc, train_loss, train_precision, train_recall, train_precision_n, train_recall_n, _, _, __, train_roc_auc = test(model_mc,[optimizer_cl,optimizer_mo],train_loader_mc,criterion_mc)
    train_metrics["acc"].append(train_acc)
    train_metrics["loss"].append(train_loss)
    train_metrics["precision"].append(train_precision)
    train_metrics["recall"].append(train_recall)
    train_metrics["roc_auc"].append(train_roc_auc)
    print("BEFORE TEST(VAL_LOADER)")
    val_acc, val_loss, val_precision, val_recall, val_precision_n, val_recall_n, _, _, __, val_roc_auc = test(model_mc,[optimizer_cl,optimizer_mo],val_loader_mc,criterion_mc)
    if epoch==0 or val_roc_auc>np.max(val_metrics["roc_auc"]):
        PATH = os.path.join(outdir,'model_best_auc.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_mo.state_dict(),
#             'loss': loss,
            }, PATH)
        PATH = os.path.join(outdir,'classifier_best_auc.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer_cl.state_dict(),
#             'loss': loss,
            }, PATH)
        PATH = os.path.join(outdir,'discriminator_best_auc.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_di.state_dict(),
#             'loss': loss,
            }, PATH)
    val_metrics["acc"].append(val_acc)
    val_metrics["loss"].append(val_loss)
    val_metrics["precision"].append(val_precision)
    val_metrics["recall"].append(val_recall)
    val_metrics["roc_auc"].append(val_roc_auc)
    print("Epoch ",epoch," Train acc: ",train_acc," loss: ",train_loss," precision: ",train_precision," recall: ",train_recall)#," precision_n: ",train_precision_n," recall_n: ",train_recall_n)
    print("Epoch ",epoch," Val   acc: ",val_acc,  " loss: ",val_loss," precision: ",val_precision," recall: ",val_recall)#," precision_n: ",val_precision_n," recall_n: ",val_recall_n)

    # EVALUATE THE DISCRIMINATOR PERFORMANCE
    train_acc, train_loss, train_precision, train_recall, train_precision_n, train_recall_n, _, _, __, train_roc_auc = test_discr(model_dt,[optimizer_mo,optimizer_di],train_loader_dt,criterion_dt)
    train_metrics_da["acc"].append(train_acc)
    train_metrics_da["loss"].append(train_loss)
    train_metrics_da["precision"].append(train_precision)
    train_metrics_da["recall"].append(train_recall)
    train_metrics_da["roc_auc"].append(train_roc_auc)

    val_acc, val_loss, val_precision, val_recall, val_precision_n, val_recall_n, _, _, __, val_roc_auc = test_discr(model_dt,[optimizer_mo,optimizer_di],val_loader_dt,criterion_dt)
    val_metrics_da["acc"].append(val_acc)
    val_metrics_da["loss"].append(val_loss)
    val_metrics_da["precision"].append(val_precision)
    val_metrics_da["recall"].append(val_recall)
    val_metrics_da["roc_auc"].append(val_roc_auc)

    print("DISCRIMINATOR: Epoch ",epoch," Train acc: ",train_acc," loss: ",train_loss," precision: ",train_precision," recall: ",train_recall)#," precision_n: ",train_precision_n," recall_n: ",train_recall_n)
    print("DISCRIMINATOR: Epoch ",epoch," Val   acc: ",val_acc,  " loss: ",val_loss," precision: ",val_precision," recall: ",val_recall)#," precision_n: ",val_precision_n," recall_n: ",val_recall_n)

# for name, param in model.named_parameters():
#     if 'weight' in name:
#         print("DEBUGGING: name, param = ",name,param)
#     break
import matplotlib.pyplot as plt
epochs = [i for i in range(len(train_metrics["loss"]))]
f = plt.figure()
ymax = max(max(train_metrics['loss']),max(val_metrics['loss']))
ymax = 0.01 #max(max(train_metrics['loss']),max(val_metrics['loss']))
print("ymax = ",ymax)
plt.ylim((10**-3,10**-1))
plt.semilogy(epochs, train_metrics['loss'], color='tab:orange', linewidth=2, markersize=1, label="Training")
plt.semilogy(epochs, val_metrics['loss'], color='tab:blue', linewidth=2, markersize=1, label="Validation")
plt.semilogy(epochs, train_metrics_da['loss'], color='tab:pink', linewidth=2, markersize=1, label="Discr. Training")
plt.semilogy(epochs, val_metrics_da['loss'], color='tab:olive', linewidth=2, markersize=1, label="Discr. Validation")
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Loss')
f.savefig(os.path.join(outdir,'loss_da.pdf'))
plt.close()

f = plt.figure()
plt.ylim(1e-5,1)
plt.plot(epochs, train_metrics['acc'], color='tab:orange', linewidth=2, markersize=1, label="Training")
plt.plot(epochs, val_metrics['acc'], color='tab:blue', linewidth=2, markersize=1, label="Validation")
plt.semilogy(epochs, train_metrics_da['acc'], color='tab:pink', linewidth=2, markersize=1, label="Discr. Training")
plt.semilogy(epochs, val_metrics_da['acc'], color='tab:olive', linewidth=2, markersize=1, label="Discr. Validation")
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
f.savefig(os.path.join(outdir,'acc_da.pdf'))
plt.close()

f = plt.figure()
plt.ylim(0,1)
plt.plot(epochs, train_metrics['precision'], color='tab:orange', linewidth=2, markersize=1, label="Training")
plt.plot(epochs, val_metrics['precision'], color='tab:blue', linewidth=2, markersize=1, label="Validation")
plt.semilogy(epochs, train_metrics_da['precision'], color='tab:pink', linewidth=2, markersize=1, label="Discr. Training")
plt.semilogy(epochs, val_metrics_da['precision'], color='tab:olive', linewidth=2, markersize=1, label="Discr. Validation")
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Precision')
f.savefig(os.path.join(outdir,'prec_discr_da.pdf'))
plt.close()

f = plt.figure()
plt.ylim(0,1)
plt.plot(epochs, train_metrics['recall'], color='tab:orange', linewidth=2, markersize=1, label="Training")
plt.plot(epochs, val_metrics['recall'], color='tab:blue', linewidth=2, markersize=1, label="Validation")
plt.semilogy(epochs, train_metrics_da['recall'], color='tab:pink', linewidth=2, markersize=1, label="Discr. Training")
plt.semilogy(epochs, val_metrics_da['recall'], color='tab:olive', linewidth=2, markersize=1, label="Discr. Validation")
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Recall')
f.savefig(os.path.join(outdir,'recall_da.pdf'))
plt.close()

#NOTE: ALSO TRY CATBOOST WITH NEW VERTEXING
#NOTE: ALSO TRY GIN/PNet WITH NEW VERTEXING
#NOTE: SEE BELOW FIRST: CAN ALSO IMPLEMENT IN PL SO YOU CAN USE MULTIPLE GPUs
#NOTE: THIS WILL TAKE ROUGHLY 21min for 497000 training samples without GPU so just install GPU version of pytorch and save time.

# Save logs to yaml
import yaml

with open(os.path.join(outdir,'train_metrics.yaml'), 'w') as file:
    yaml.dump(train_metrics, file)
with open(os.path.join(outdir,'val_metrics.yaml'), 'w') as file:
    yaml.dump(val_metrics, file)

with open(os.path.join(outdir,'train_metrics_da.yaml'), 'w') as file:
    yaml.dump(train_metrics_da, file)
with open(os.path.join(outdir,'val_metrics_da.yaml'), 'w') as file:
    yaml.dump(val_metrics_da, file)

train_dataset_mc, val_dataset_mc, test_dataset_mc = datasets_mc
train_dataset_dt, val_dataset_dt, test_dataset_dt = datasets_both

#---------- NOW RUN ON TEST DATASETS ----------#

train_acc, train_loss, train_precision, train_recall, train_precision_n, train_recall_n, outputs, decisions, y_true, roc_auc_test = test(model_mc,[optimizer_cl,optimizer_mo],test_loader_mc,criterion_mc)
import sklearn
import seaborn as sns
cm = confusion_matrix(y_true,decisions,labels=[0,1])

#TODO: PLOT ROC CURVE
"""
TODO: Add modeloss to see if that gets rid of BG sculpting.
TODO: Implement C++/Java loading -> Make available as library function to call in your C++/Java analysis

"""

tp = cm[1,1]
fp = cm[0,1] # bg classified as signal is 0,1
fn = cm[1,0] # signal classified as bg  is 1,0
tn = cm[0,0]
        
precision = tp / (tp + fp) if tp+fp>0 else 0
recall = tp / (tp + fn) if tp+fn>0 else 0
precision_n = tn / (tn + fn) if tn+fn>0 else 0
recall_n = tn / (tn + fp) if tn+fp>0 else 0
precision2 = sklearn.metrics.precision_score(y_true, decisions)
recall2 = sklearn.metrics.recall_score(y_true,decisions)
print("precision2 = ",precision2)
print("precision = ",precision)
print("recall2 = ",recall2)
print("recall = ",recall)
print("precision_n = ",precision_n)
print("recall_n = ",recall_n)
print("DEBUGGING: roc_auc_test = ",roc_auc_test)

def plot_matrix(cm, classes, title):
    ax = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
    ax.set(title=title, xlabel="predicted label", ylabel="true label")
    
classes = ['bg', 'sig']
title = "title example"

plot_matrix(cm, classes, title)

# Get separated network output arrays
print("DEBUGGING: len(outputs)    = ",len(outputs))
print("DEBUGGING: outputs[0:10]   = ",outputs[0:10])
print("DEBUGGING: decisions[0:10] = ",decisions[0:10])
print("DEBUGGING: y_true[0:10]    = ",y_true[0:10])
outputs   = np.array([el.tolist() for el in outputs])
decisions = np.array([el.item() for el in decisions])
y_true    = np.array([el.item() for el in y_true])
print("DEBUGGING: AFTER CREATING np.arrays")
print("DEBUGGING: outputs[0:10]   = ",outputs[0:10])
print("DEBUGGING: decisions[0:10] = ",decisions[0:10])
print("DEBUGGING: y_true[0:10]    = ",y_true[0:10])
print("DEBUGGING: type(outputs)       = ",type(outputs))
print("DEBUGGING: type(decisions)     = ",type(decisions))
print("DEBUGGING: type(y_true)        = ",type(y_true))
print("DEBUGGING: np.shape(outputs)   = ",np.shape(outputs))
print("DEBUGGING: np.shape(decisions) = ",np.shape(decisions))
print("DEBUGGING: np.shape(y_true)    = ",np.shape(y_true))
print("DEBUGGING: np.shape(decisions==1) = ",np.shape(decisions==1))
outputs_sig_true  = outputs[:,1][np.logical_and(decisions==1,y_true==1)]
outputs_sig_false = outputs[:,1][np.logical_and(decisions==1,y_true==0)]
outputs_bg_false  = outputs[:,1][np.logical_and(decisions==0,y_true==1)]
outputs_bg_true   = outputs[:,1][np.logical_and(decisions==0,y_true==0)]

l_sig_true = []
l_sig_false = []
l_bg_false = []
l_bg_true = []
k_sig_true = []
k_sig_false = []
k_bg_false = []
k_bg_true = []
for i, data in enumerate(test_dataset_mc):
    decision = decisions[i]
    y = y_true[i]
    kin = data.kinematics.tolist()
#     if len(kin)!=13: raise TypeError
    if decision==1 and decision==y:
        l_sig_true.append(data)
        k_sig_true.append(kin[0])#NOTE: Should add matching indices specification to data objects...
    elif decision==1 and decision!=y:
        l_sig_false.append(data)
        k_sig_false.append(kin[0])
    elif decision==0 and decision!=y:
        l_bg_false.append(data)
        k_bg_false.append(kin[0])
    elif decision==0 and decision==y:
        l_bg_true.append(data)
        k_bg_true.append(kin[0])
b_sig_true = torch_geometric.data.Batch().from_data_list(l_sig_true)
b_sig_false = torch_geometric.data.Batch().from_data_list(l_sig_false)
b_bg_false = torch_geometric.data.Batch().from_data_list(l_bg_false)
b_bg_true = torch_geometric.data.Batch().from_data_list(l_bg_true)
print(b_sig.x.shape)
print(b_sig.y.shape)
print(b_bg.x.shape)
print(b_bg.y.shape)



def plot_data_separated(array_sig_true,array_sig_false,array_bg_false,array_bg_true,title=None,xlabel='index',nbins=50,low=-1.1,high=1.1,logy=False):
    
    array_sig_true = array_sig_true.flatten()
    array_sig_false = array_sig_false.flatten()
    array_bg_false = array_bg_false.flatten()
    array_bg_true = array_bg_true.flatten()
    
    # Plot SIG ONLY distributions
    f = plt.figure()
    if title != None:
        plt.title(title)
    plt.title('Separated distribution MC-matched')
    plt.hist(array_sig_true, color='tab:red', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='sig true')
    plt.hist(array_sig_false, color='tab:orange', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='sig false')
#     plt.hist(array_bg_false, color='tab:green', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='bg false')
#     plt.hist(array_bg_true, color='tab:blue', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='bg true')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel(xlabel)
    if logy: plt.yscale('log')
    f.savefig(os.path.join(outdir,xlabel+'__mctest__sg.pdf'))
#     f.savefig(xlabel+'_separated_'+todays_date+'.pdf')

    # Plot SIG AND BG distributions
    f = plt.figure()
    if title != None:
        plt.title(title)
    plt.title('Separated distribution MC-matched')
    plt.hist(array_sig_true, color='tab:red', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='sig true')
    plt.hist(array_sig_false, color='tab:orange', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='sig false')
    plt.hist(array_bg_false, color='tab:green', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='bg false')
    plt.hist(array_bg_true, color='tab:blue', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='bg true')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel(xlabel)
    if logy: plt.yscale('log')
    f.savefig(os.path.join(outdir,xlabel+'__mctest__sg_bg.pdf'))
#     f.savefig(xlabel+'_separated_'+todays_date+'.pdf')
    
arr1 = b_sig_true.x
arr2 = b_sig_false.x
arr3 = b_bg_false.x
arr4 = b_bg_true.x
    
# # Plot data separated distributions
# plot_data_separated(arr1[:,0],arr2[:,0],arr3[:,0],arr4[:,0],xlabel="pT")
# plot_data_separated(arr1[:,1],arr2[:,1],arr3[:,1],arr4[:,1],xlabel="phi")
# plot_data_separated(arr1[:,2],arr2[:,2],arr3[:,2],arr4[:,2],xlabel="theta")
# plot_data_separated(arr1[:,3],arr2[:,3],arr3[:,3],arr4[:,3],xlabel="beta")
# plot_data_separated(arr1[:,4],arr2[:,4],arr3[:,4],arr4[:,4],xlabel="chi2")
# plot_data_separated(arr1[:,5],arr2[:,5],arr3[:,5],arr4[:,5],xlabel="pid")
# plot_data_separated(arr1[:,6],arr2[:,6],arr3[:,6],arr4[:,6],xlabel="status")

print(len(k_sig_true))
print(len(k_sig_true[1]))
arr1 = np.array(k_sig_true)
arr2 = np.array(k_sig_false)
arr3 = np.array(k_bg_false)
arr4 = np.array(k_bg_true)
print(type(arr1))
print(type(arr1[0]))
print(arr1.shape)
print(arr1[0].shape)
#mass_index, Q2_index, W_index, x_index, y_index, z_index, xF_index, mc_pid_pa_p_index, mc_pid_ppa_p_index, mc_idx_pa_p_index, mc_idx_ppa_p_index, mc_idx_pa_pim_index, mc_label_index
# Plot data separated distributions
plot_data_separated(outputs_sig_true,outputs_sig_false,outputs_bg_false,outputs_bg_true,xlabel="GNN Output Probability",low=0.0,high=1.0,logy=True)
plot_data_separated(arr1[:,10],arr2[:,10],arr3[:,10],arr4[:,10],xlabel="mass_ppim",low=1.08,high=1.24)
plot_data_separated(arr1[:,4],arr2[:,4],arr3[:,4],arr4[:,4],xlabel="Q2",low=0.0,high=8.0)
plot_data_separated(arr1[:,5],arr2[:,5],arr3[:,5],arr4[:,5],xlabel="W",low=0.0,high=8.0)
plot_data_separated(arr1[:,6],arr2[:,6],arr3[:,6],arr4[:,6],xlabel="x",low=0.0,high=1.0)
plot_data_separated(arr1[:,7],arr2[:,7],arr3[:,7],arr4[:,7],xlabel="y",low=0.0,high=1.0)
plot_data_separated(arr1[:,8],arr2[:,8],arr3[:,8],arr4[:,8],xlabel="z_ppim",low=0.0,high=1.5)
plot_data_separated(arr1[:,9],arr2[:,9],arr3[:,9],arr4[:,9],xlabel="xF_ppim",low=-2.0,high=2.0)

#--------- NOW TEST ON DATA ----------#
@torch.no_grad()
def data_test(model,optimizer,loader,criterion):
    model.eval()
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             print("DEBUGGING: name, param = ",name,param)
    correct = 0
    loss_ = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    decisions = []
    outputs = []
    y_true = []
    for data in tqdm(loader):  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss_ += loss.item()
        out = torch.nn.functional.softmax(out,dim=-1)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        
        outputs.extend(out.cpu())
        decisions.extend(pred.cpu())

    return loss_ / len(loader.dataset), outputs, decisions # Derive ratio of correct predictions.

# acc, loss = test(train_loader)
# print(acc,loss)
# acc, loss = test(val_loader)

train_loss, outputs, decisions = data_test(model_mc,optimizer_cl,test_loader_dt,criterion_mc)

# Get separated network output arrays
print("DEBUGGING: len(outputs)    = ",len(outputs))
print("DEBUGGING: outputs[0:10]   = ",outputs[0:10])
print("DEBUGGING: decisions[0:10] = ",decisions[0:10])
outputs   = np.array([el.tolist() for el in outputs])
decisions = np.array([el.item() for el in decisions])
print("DEBUGGING: AFTER CREATING np.arrays")
print("DEBUGGING: outputs[0:10]   = ",outputs[0:10])
print("DEBUGGING: decisions[0:10] = ",decisions[0:10])
print("DEBUGGING: type(outputs)       = ",type(outputs))
print("DEBUGGING: type(decisions)     = ",type(decisions))
print("DEBUGGING: np.shape(outputs)   = ",np.shape(outputs))
print("DEBUGGING: np.shape(decisions) = ",np.shape(decisions))
print("DEBUGGING: np.shape(decisions==1) = ",np.shape(decisions==1))
outputs_sig  = outputs[:,1][decisions==1]
outputs_bg   = outputs[:,1][decisions==0]

l_sig = []
l_bg = []
k_sig = []
k_bg = []
for i, data in enumerate(test_dataset_dt):
    decision = decisions[i]
#     y = y_true[i]
    kin = data.kinematics.tolist()
#     if len(kin)!=13: raise TypeError
    if decision==1:
        l_sig.append(data)
        k_sig.append(kin[0])#NOTE: Should add matching indices specification to data objects...
    if decision==0:
        l_bg.append(data)
        k_bg.append(kin[0])
        
b_sig = torch_geometric.data.Batch().from_data_list(l_sig)
b_bg  = torch_geometric.data.Batch().from_data_list(l_bg)
print(b_sig.x.shape)
print(b_sig.y.shape)
print(b_bg.x.shape)
print(b_bg.y.shape)

def plot_data_separated(array_sig,array_bg,title=None,xlabel='index',nbins=50,low=-1.1,high=1.1,logy=False):
    
    array_sig = array_sig.flatten()
    array_bg = array_bg.flatten()
    
    # Plot SIG ONLY distributions
    f = plt.figure()
    if title != None:
        plt.title(title)
    plt.title('Separated distribution MC-matched')
    plt.hist(array_sig, color='tab:orange', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='sig')
#     plt.hist(array_bg, color='tab:blue', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='bg')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel(xlabel)
    if logy: plt.yscale('log')
    f.savefig(os.path.join(outdir,xlabel+'__dttest__sg.pdf'))
#     f.savefig(xlabel+'_data_separated_'+todays_date+'.pdf')

    # Plot SIG and BG distributions
    f = plt.figure()
    if title != None:
        plt.title(title)
    plt.title('Separated distribution MC-matched')
    plt.hist(array_sig, color='tab:orange', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='sig')
    plt.hist(array_bg, color='tab:blue', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='bg')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel(xlabel)
    if logy: plt.yscale('log')
    f.savefig(os.path.join(outdir,xlabel+'__dttest__sg_bg.pdf'))
#     f.savefig(xlabel+'_data_separated_'+todays_date+'.pdf')
    
arr1 = b_sig.x
arr2 = b_bg.x
    
# # Plot data separated distributions
# plot_data_separated(arr1[:,0],arr2[:,0],xlabel="pT")
# plot_data_separated(arr1[:,1],arr2[:,1],xlabel="phi")
# plot_data_separated(arr1[:,2],arr2[:,2],xlabel="theta")
# plot_data_separated(arr1[:,3],arr2[:,3],xlabel="beta")
# plot_data_separated(arr1[:,4],arr2[:,4],xlabel="chi2")
# plot_data_separated(arr1[:,5],arr2[:,5],xlabel="pid")
# plot_data_separated(arr1[:,6],arr2[:,6],xlabel="status")

print(len(k_sig))
print(len(k_sig[1]))
arr1 = np.array(k_sig)
arr2 = np.array(k_bg)
print(type(arr1))
print(type(arr1[0]))
print(arr1.shape)
print(arr1[0].shape)
#mass_index, Q2_index, W_index, x_index, y_index, z_index, xF_index, mc_pid_pa_p_index, mc_pid_ppa_p_index, mc_idx_pa_p_index, mc_idx_ppa_p_index, mc_idx_pa_pim_index, mc_label_index
# Plot data separated distributions
plot_data_separated(outputs_sig,outputs_bg,xlabel="GNN Output Probability",low=0.0,high=1.0,logy=True)
plot_data_separated(arr1[:,10],arr2[:,10],xlabel="mass_ppim",low=1.08,high=1.24)
plot_data_separated(arr1[:,4],arr2[:,4],xlabel="Q2",low=0.0,high=8.0)
plot_data_separated(arr1[:,5],arr2[:,5],xlabel="W",low=0.0,high=8.0)
plot_data_separated(arr1[:,6],arr2[:,6],xlabel="x",low=0.0,high=1.0)
plot_data_separated(arr1[:,7],arr2[:,7],xlabel="y",low=0.0,high=1.0)
plot_data_separated(arr1[:,8],arr2[:,8],xlabel="z_ppim",low=0.0,high=1.5)
plot_data_separated(arr1[:,9],arr2[:,9],xlabel="xF_ppim",low=-2.0,high=2.0)
