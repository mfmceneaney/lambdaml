#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

import os.path as osp
from glob import glob

import torch
from torch_geometric.data import Dataset, InMemoryDataset, download_url

class CustomInMemoryDataset(InMemoryDataset):
    """
    :class: CustomInMemoryDataset
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, datalist=[], idx=0):
        self.datalist = datalist
        self.idx = idx #NOTE: This is index of batch for writing larger datasets
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'data{self.idx}.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.datalist

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])

class CustomDataset(Dataset):
    """
    :class: CustomDataset
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return sorted(glob(osp.join(self.root,'processed/')+'data*.pt')) #TODO: Set these in __init__() How to get file names from directory?

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data{idx}.pt'))
        return data
