# DATA
# pylint: disable=no-member
# pylint: disable=abstract-method
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import os
from glob import glob
import multiprocessing
from tqdm import tqdm
from functools import lru_cache
import json

# Local imports
from .log import setup_logger


# Set module logger
logger = setup_logger(__name__)


# Class definitions
class SmallDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        datalist=None,
        clean_keys=(),
    ):
        self.datalist = datalist
        self.root = root
        self.clean_keys = clean_keys
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["some_file_1", "some_file_2"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def clean_data(self, data):

        # Create a new graph and remove undesired attributes
        logger.debug("Cleaning data : %s", data)
        if not isinstance(data, Data):
            return data
        cleaned_data = Data()
        for key in data.keys():
            if key in (self.clean_keys):
                continue
            value = getattr(data, key)
            if isinstance(value, torch.Tensor):
                cleaned_data[key] = value.detach().cpu().clone()
        return cleaned_data

    def process(self):

        # Check input data list
        if self.datalist is None or len(self.datalist) == 0:
            return

        # Read data into huge `Data` list.
        data_list = self.datalist

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_list = [self.clean_data(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])

    # def get(self, idx):
    #     if self.datalist is None or len(self.datalist)==0: self.datalist = list(torch.load(os.path.join(self.processed_dir, self.processed_file_names[0])))
    #     data = self.datalist[idx]
    #     return data


class LargeDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        datalist=None,
        num_workers=8,
        chunk_size=100,
        pickle_protocol=5,
        clean_keys=("is_data", "rec_indices"),
    ):
        self.datalist = datalist
        self.root = root
        self.num_workers = num_workers
        self.pickle_protocol = pickle_protocol
        self.chunk_size = chunk_size
        self.clean_keys = clean_keys
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        if self.datalist is not None and len(self.datalist) > 0:
            return [f"data{i}.pt" for i in range(len(self.datalist))]
        else:
            return [
                os.path.basename(path)
                for path in glob(os.path.join(self.raw_dir, "*.pt"))
            ]

    @property
    def processed_file_names(self):
        if self.datalist is not None and len(self.datalist) > 0:
            return [f"data{i}.pt" for i in range(len(self.datalist))]
        else:
            return [
                os.path.basename(path)
                for path in glob(os.path.join(self.processed_dir, "*.pt"))
            ]

    def clean_data(self, data):

        # Create a new graph and remove undesired attributes
        logger.debug("Cleaning data : %s", data)
        if not isinstance(data, Data):
            return data
        cleaned_data = Data()
        for key in data.keys():
            if key in (self.clean_keys):
                continue
            value = getattr(data, key)
            if isinstance(value, torch.Tensor):
                cleaned_data[key] = value.detach().cpu().clone()
        return cleaned_data

    def save_graph(self, idx):

        # Select data
        data = self.datalist[idx]

        # Apply filters and transforms
        if self.pre_filter is not None and not self.pre_filter(data):
            return
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save data
        torch.save(
            self.clean_data(data),
            os.path.join(self.processed_dir, self.processed_file_names[idx]),
            pickle_protocol=self.pickle_protocol,
        )

    def process(self):

        # Check input data list
        if self.datalist is None or len(self.datalist) == 0:
            return

        # Save graphs in several processes
        with multiprocessing.Pool(
            processes=min(len(self.datalist), self.num_workers)
        ) as pool:
            try:
                list(
                    tqdm(
                        pool.imap_unordered(
                            self.save_graph, range(len(self.datalist)), self.chunk_size
                        ),
                        total=len(self.datalist),
                    )
                )
            except KeyboardInterrupt as e:
                print("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
                pool.join()
                print(e)
            else:
                pool.close()
                pool.join()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            os.path.join(self.processed_dir, self.processed_file_names[idx])
        )
        return data


@lru_cache(maxsize=16)
def load_batch(path, weights_only=True):
    return list(torch.load(path, weights_only=weights_only))


class LazyDataset(Dataset):

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        datalist=None,
        num_workers=0,
        chunk_size=100,
        pickle_protocol=5,
        clean_keys=("is_data", "rec_indices"),
        batch_size=100000,
        drop_last=False,
        weights_only=False,
        recreate=False,
    ):
        self.metadata_file_name = "metadata.json"
        self.datalist = datalist
        self.root = root
        self.num_workers = num_workers
        self.pickle_protocol = pickle_protocol
        self.chunk_size = chunk_size
        self.clean_keys = clean_keys
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_batches = 0
        if self.datalist is not None and len(self.datalist) > 0 and self.batch_size > 0:
            self.num_batches = len(self.datalist) // self.batch_size + (
                1
                if len(self.datalist) % self.batch_size > 0 and not self.drop_last
                else 0
            )
        self.size = (
            len(datalist) - (len(datalist) % self.batch_size if self.drop_last else 0)
            if datalist is not None
            else 0
        )
        self.weights_only = weights_only
        self.process_batch_start_idx = 0

        # Check if metadata already exists
        metadata_path = os.path.join(self.root, self.metadata_file_name)
        metadata = {
            "size": self.size,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
        }
        if not recreate and os.path.exists(metadata_path):

            # First try opening file and reading metadata
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:

                    # Read metadata from previously saved dataset
                    metadata = json.load(f)
                    self.size += metadata["size"]  # NOTE: Ordering matters here.
                    self.batch_size = metadata["batch_size"]
                    self.num_batches = self.size // self.batch_size + (
                        1
                        if self.size % self.batch_size > 0 and not self.drop_last
                        else 0
                    )
                    self.process_batch_start_idx = (
                        metadata["num_batches"] - 1
                    )  # NOTE: This will be incremented below if the last batch is full.

                    # Check that the data files and the number of batches for the full dataset
                    # without dropping the last batch are consistent
                    actual_data_files = sorted(glob(os.path.join(self.processed_dir, "data*.pt")))
                    expect_data_files = sorted([f"data{i}.pt" for i in range(metadata['num_batches'])])
                    if not np.all(actual_datafiles==expect_data_files):
                        raise RuntimeError(
                            f"Number of data files {len(actual_datafiles)} does not match number of batches {metadata['num_batches']}, dataset is corrupted!"
                        )

                    # Reset metadata for when you write it below
                    metadata = {
                        "size": self.size,
                        "batch_size": self.batch_size,
                        "num_batches": self.num_batches,
                    }

            except FileNotFoundError as e:
                print("Metadata file not found, dataset is corrupted!")
                raise e

            # Load last batch and check if length is same as batch size.
            # If it is not, then prepend this batch to self.datalist so that everything gets saved correctly.
            # It is important to use torch.load and not load_batch here so that you don't end up loading
            # a partial batch later on!
            _loaded_batch = torch.load(
                os.path.join(
                    self.processed_dir,
                    self.processed_file_names[self.process_batch_start_idx],
                ),
                weights_only=self.weights_only,
            )
            if len(_loaded_batch) != self.batch_size:
                if self.datalist is None:
                    self.datalist = []
                self.datalist = [*_loaded_batch, *self.datalist]
            else:
                self.process_batch_start_idx += 1  # NOTE: If last batch is full, increment the starting batch index.

        # (Re)create the metadata file
        os.makedirs(self.root, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
            return [f"data{i}.pt" for i in range(self.num_batches)]

    @property
    def processed_file_names(self):
            return [f"data{i}.pt" for i in range(self.num_batches)]

    def clean_data(self, data):

        # Create a new graph and remove undesired attributes
        logger.debug("Cleaning data : %s", data)
        if not isinstance(data, Data):
            return data
        cleaned_data = Data()
        for key in data.keys():
            if key in (self.clean_keys):
                continue
            value = getattr(data, key)
            if isinstance(value, torch.Tensor):
                cleaned_data[key] = value.detach().cpu().clone()
        return cleaned_data

    def save_graph_batch(self, idx):

        min_idx = idx * self.batch_size
        max_idx = min((idx + 1) * self.batch_size, len(self.datalist))

        data = self.datalist[min_idx:max_idx]

        if self.pre_filter is not None:
            data = [d for d in data if not self.pre_filter(data)]

        if self.pre_transform is not None:
            data = [self.pre_transform(d) for d in data]

        data = [self.clean_data(d) for d in data]

        logger.info(
            "Saving batch %d with %d graphs",
            idx + self.process_batch_start_idx,
            len(data),
        )
        logger.info(
            "Processed dir: %s  File name: %s",
            self.processed_dir,
            self.processed_file_names[idx + self.process_batch_start_idx],
        )
        torch.save(
            data,
            os.path.join(
                self.processed_dir,
                self.processed_file_names[idx + self.process_batch_start_idx],
            ),
            pickle_protocol=self.pickle_protocol,
        )

    def process(self):

        # Check input data list
        if self.datalist is None or len(self.datalist) == 0:
            return

        local_num_batches = self.num_batches - self.process_batch_start_idx

        if self.num_workers <= 0:
            if local_num_batches > 1:
                tqdm([self.save_graph_batch(idx) for idx in range(local_num_batches)])
            else:
                self.save_graph_batch(0)
        else:
            with multiprocessing.Pool(
                processes=min(local_num_batches, self.num_workers)
            ) as pool:
                try:
                    list(
                        tqdm(
                            pool.imap_unordered(
                                self.save_graph_batch,
                                range(local_num_batches),
                                self.chunk_size,
                            ),
                            total=local_num_batches,
                        )
                    )
                except KeyboardInterrupt as e:
                    print("Caught KeyBoardInterrupt, terminating workers")
                    pool.terminate()
                    pool.join()
                    print(e)
                else:
                    pool.close()
                    pool.join()

    def len(self):
        return self.size

    def get(self, idx):

        # Check index
        if idx >= self.size:
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        # Get indices
        batch_idx = idx // self.batch_size
        within_idx = idx % self.batch_size

        # Load batch data and select at index
        _loaded_batch = load_batch(
            os.path.join(self.processed_dir, self.processed_file_names[batch_idx]),
            weights_only=self.weights_only,
        )
        logger.debug("len(_loaded_batch) = %d", len(_loaded_batch))
        logger.debug("within_idx = %d", within_idx)
        data = _loaded_batch[within_idx]
        return data


def get_sampler_weights(ds):
    """
    :params:
        ds : Dataset

    :return:
        sampler weights

    :description:
        Given a labelled dataset, generate a list of weights for a sampler such that all classes are equally probable.
    """

    # Count unique labels and weight them inversely to their total counts
    labels = torch.tensor([data.y.item() for data in ds])
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    sampler_weights = [class_weights[label] for label in labels]
    return sampler_weights
