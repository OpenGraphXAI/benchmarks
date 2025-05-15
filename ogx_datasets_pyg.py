import os
import math
import torch
import random
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
from collections import defaultdict
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset, Data, extract_zip
from torch_geometric.data.download import download_url

class OGXBenchmark(InMemoryDataset):
    def __init__(self, root = './data', data_name='alfa', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'OGXBenchmark'
        self.data_name = data_name
        self.identifier = f'{self.name}_{self.data_name}'
        self.url = f'https://drive.usercontent.google.com/download?id=1-1cqTlzQLXBaKLrsmdmOvtomS_6-kdCP&confirm=t'
        random.seed(54398724)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.stratification_labels = self.load_stratification_labels()
        self.mapper = self.load_mapper()
        self.splits = self.load_splits()
        self.explanations = self.load_explanations()

    def download(self):
        path = download_url(self.url, self.root, True, 'file.zip')
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    @property
    def raw_file_names(self):
        return [f'{self.data_name}.pkl', f'{self.data_name}_splits.pkl']

    @property
    def processed_file_names(self):
        return [f'{self.identifier}.pt', f'{self.identifier}_strat_labels.pkl', f'{self.identifier}_mapper.pkl', f'{self.identifier}_gts.pkl']

    def process(self):
        with open(self.raw_paths[0],'rb') as fin:
            raws = pkl.load(fin)

        c0 = [from_networkx(graph, group_node_attrs=['x']) for graph in raws['class0']]
        c1 = [from_networkx(graph, group_node_attrs=['x']) for graph in raws['class1']]

        targets = [0]*len(c0) + [1]*len(c1)
        data = c0 + c1

        zipped = list(zip(data, targets))
        random.shuffle(zipped)
        data, targets = zip(*zipped)

        explanations = []

        data_list= []
        for idx, (elem, target) in enumerate(zip(data, targets)):
            attributes = elem.keys

            ex2 = []
            if 'mask' in attributes:
                ego_mask = elem.mask.int()
                ex2.append(ego_mask)

            explanations.append(ex2)

            for attr in attributes:
                if attr != 'edge_index' and attr != 'x' and attr != 'smile':
                    del elem[attr]
                    
            elem.y = torch.LongTensor([target])
            data_list.append(elem)

        atn = set()
        for elem in data_list:
            for an in elem.x[:,0]:
                atn.add(an.item())
        atn = list(atn)

        mapperAtn = {}
        for i, at in enumerate(atn):
            mapperAtn[at] = i

        for k, data in enumerate(data_list):
            indexes = torch.tensor([mapperAtn[x.item()] for x in data.x[:,0]]).view(-1,1)
            ohAtm = torch.nn.functional.one_hot(indexes[:,0].to(torch.int64), len(mapperAtn))
            data_list[k].x = ohAtm.float()

        nns=[]
        for data in data_list:
            nns.append(data.num_nodes)

        df = pd.DataFrame({'nns': nns})
        df["nns_binned"] = pd.qcut(df["nns"], q=5)
        df["nns_binned"] = df["nns_binned"].cat.codes
        stratifying_label = df["nns_binned"].to_numpy().astype(str)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        with open(self.processed_paths[1], 'wb') as file:
            pkl.dump(stratifying_label, file)

        with open(self.processed_paths[2], 'wb') as file:
            pkl.dump(mapperAtn, file)

        with open(self.processed_paths[3], 'wb') as file:
            pkl.dump(explanations, file)

    def load_stratification_labels(self):
        with open(self.processed_paths[1], 'rb') as file:
            return pkl.load(file)

    def load_mapper(self):
        with open(self.processed_paths[2], 'rb') as file:
            return pkl.load(file)

    def load_explanations(self):
        with open(self.processed_paths[3], 'rb') as file:
            return pkl.load(file)

    def load_splits(self):
        with open(self.raw_paths[1],'rb') as file:
            return pkl.load(file)[0]

if __name__ == "__main__":
    dataset = OGXBenchmark(data_name='bravo')