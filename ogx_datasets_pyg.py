import os
import pickle as pkl
from typing import Callable, Optional

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.utils import from_networkx


class OGXBenchmark(InMemoryDataset):
    """
    OpenGraphXAI benchmark datasets.

    :param root: Root directory where the dataset should be saved.
    :param name: The name of the dataset (e.g. 'alfa').
    :param split: Dataset split ('train', 'val', 'test'). If not specified, the entire dataset is loaded.
    :param transform: A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before every access.
    :param pre_transform: A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before being saved to disk.
    :param pre_filter: A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the final dataset.
    :param force_reload: Whether to re-process the dataset.
    """

    url = r'https://github.com/OpenGraphXAI/benchmarks/raw/refs/heads/main/data/raw/'

    def __init__(self,
                 root: str,
                 name: str,
                 split: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):

        assert split in ['train', 'val', 'test'] or split is None, f'Unknown split: "{split}"'

        self.name_id = name
        self.name = f'OGX_{self.name_id}'

        super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
                         force_reload=force_reload)

        if split == 'train':
            self.load(self.processed_paths[1])
        elif split == 'val':
            self.load(self.processed_paths[2])
        elif split == 'test':
            self.load(self.processed_paths[3])
        else:
            self.load(self.processed_paths[0])

    def download(self):
        for raw_file in self.raw_file_names:
            download_url(f'{self.url}{raw_file}', self.raw_dir)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.name_id}.pkl', f'{self.name_id}_splits.pkl']

    @property
    def processed_file_names(self):
        return [f'{self.name}.pt'] + [f'{self.name}_{split}.pt' for split in ['train', 'val', 'test']]

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            graphs = pkl.load(f)

        with open(self.raw_paths[1], 'rb') as f:
            splits = pkl.load(f)

        data_list = []

        for class_idx in (0, 1):
            for graph in graphs[f'class{class_idx}']:
                data = from_networkx(graph)
                data_list.append(Data(x=data.x,
                                      edge_index=data.edge_index,
                                      mask=data.mask if hasattr(data, 'mask') else torch.zeros_like(data.x).bool(),
                                      mask_root=data.mask_root if hasattr(data, 'mask_root') else torch.zeros_like(data.x).bool(),
                                      y=torch.tensor([class_idx])))

        for i, split in enumerate(['train', 'val', 'test']):
            split_data = [data_list[idx] for idx in splits[0][split]]
            if self.pre_filter is not None:
                split_data = [data for data in split_data if self.pre_filter(data)]
            if self.pre_transform is not None:
                split_data = [self.pre_transform(data) for data in split_data]
            self.save(split_data, self.processed_paths[i + 1])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}({len(self)})'
