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

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, datalist=[], idx=0, processed_file_name=None):
        self.datalist = datalist
        self.idx = idx #NOTE: This is index of batch for writing larger datasets
        self.processed_file_name = processed_file_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'data{self.idx}.pt'] if self.processed_file_name is None else [self.processed_file_name]

    def process(self):
        # Read data into huge `Data` list.
        if self.datalist is None or len(self.datalist)==0: return
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

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, max_files=0):
        self.length = 0 # Overall dataset length
        self.lengths = [0] # List of cumulative data file lengths
        self.current = 0 # Current data file index
        self.current_ds = None # CustomInMemoryDataset object of current data file
        self.max_files = max_files
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        pfns = sorted(glob(osp.join(self.root,'processed/')+'data*.pt'))
        return pfns if self.max_files<=0 else pfns[:max_files]

    def len(self):
        # Loop through all dataset files loading as in memory datasets and add lengths
        if self.length>0: return self.length
        for idx_pfn, pfn in enumerate(self.processed_file_names):
            current_ds = CustomInMemoryDataset(
                            self.root,
                            transform=None,
                            pre_transform=None,
                            pre_filter=None,
                            datalist=[],
                            idx=idx_pfn,
                            processed_file_name=osp.basename(pfn),
                        )
            self.length += len(current_ds)
            self.lengths.append(self.length)
        return self.length

    def get(self, idx): #NOTE THIS SHOULD GIVE YOU A SINGLE GRAPH!

        nevents = len(self)#NOTE: NEED TO ENSURE THIS IS CALLED BEFORE LOOPING!

        # For quick looping since you're in the same data file or next data file most of the time do this
        for _ in self.processed_file_names:
            if idx>=self.lengths[self.current] and idx<self.lengths[self.current+1]:
                if self.current_ds is None or self.current!=self.current_ds.idx:
                    self.current_ds = CustomInMemoryDataset(
                            self.root,
                            transform=None,
                            pre_transform=None,
                            pre_filter=None,
                            datalist=[],
                            idx=self.current,
                            processed_file_name=osp.basename(self.processed_file_names[self.current]),
                        )
                return self.current_ds[idx-self.lengths[self.current]]
            else: self.current = (self.current+1)%len(self.processed_file_names)
        raise IndexError
