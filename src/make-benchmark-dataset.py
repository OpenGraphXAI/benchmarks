import argparse
import itertools
import json
import os.path
import pickle
from functools import reduce

from networkx import set_node_attributes, node_link_data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from graph_histogram import weisfeiler_lehman_graph_histogram, get_wl_label_nodes, get_wl_label_subgraphs


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name from TU collection', required=True)
parser.add_argument('--root', help='root directory for source dataset', default='/tmp')
parser.add_argument('--wl', help='WL coloring iterations', type=int, default=5)
parser.add_argument('--hash-size', help='has digest size for WL labels', type=int, default=16)
parser.add_argument('--label0', help='WL label for class 0', default=None)
parser.add_argument('--label1', help='WL label for class 1', default=None)
parser.add_argument('--format', help='output format of the dataset (networkx pickle or json files)', choices=['networkx', 'json'], default='networkx')
parser.add_argument('--name', help='name for the new XAI benchmark', required=True)
parser.add_argument('--out', help='output directory', default='.')
args = parser.parse_args()

assert any((args.label0, args.label1)), 'At least one of label0 and label1 must be specified'


# Load source dataset from TU collection

if args.dataset.startswith('Tox21_'):
    dataset = itertools.chain(TUDataset(root='/tmp', name=f'{args.dataset}_training'),
                              TUDataset(root='/tmp', name=f'{args.dataset}_evaluation'),
                              TUDataset(root='/tmp', name=f'{args.dataset}_testing'))
else:
    dataset = TUDataset(root='/tmp', name=args.dataset)


# Make WL histograms

graphs, targets = [], []
for data in dataset:
    if hasattr(data, 'x') and data.x is not None:
        data.x = data.x.argmax(dim=-1)
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr = data.edge_attr.argmax(dim=-1)
    graphs.append(to_networkx(data,
                              node_attrs=['x'] if hasattr(data, 'x') and data.x is not None else None,
                              to_undirected=True))
    targets.append(data.y.item())

all_histograms = [weisfeiler_lehman_graph_histogram(graph, node_label='x', iterations=args.wl, digest_size=args.hash_size, set_wl_labels=True) for graph in tqdm(graphs, desc=f'{args.dataset} encoding')]


# Generate subsets

if args.label0 is not None and args.label1 is not None:
    subset0 = [idx for idx, (histogram, y) in enumerate(zip(all_histograms, targets)) if args.label0 in histogram and args.label1 not in histogram and y == 0]
    subset1 = [idx for idx, (histogram, y) in enumerate(zip(all_histograms, targets)) if args.label0 not in histogram and args.label1 in histogram and y == 1]
elif args.label0 is None and args.label1 is not None:
    subset0 = [idx for idx, (histogram, y) in enumerate(zip(all_histograms, targets)) if args.label1 not in histogram and y == 0]
    subset1 = [idx for idx, (histogram, y) in enumerate(zip(all_histograms, targets)) if args.label1 in histogram and y == 1]
elif args.label0 is not None and args.label1 is None:
    subset0 = [idx for idx, (histogram, y) in enumerate(zip(all_histograms, targets)) if args.label0 in histogram and y == 0]
    subset1 = [idx for idx, (histogram, y) in enumerate(zip(all_histograms, targets)) if args.label0 not in histogram and y == 1]


# Generate ground truth masks

graphs0 = [graphs[idx] for idx in subset0]
graphs1 = [graphs[idx] for idx in subset1]

if args.label0 is not None:
    for graph in graphs0:
        root_nodes = set(node for node, _ in get_wl_label_nodes(graph, args.label0))
        root_mask = {node: node in root_nodes for node in graph.nodes}
        set_node_attributes(graph, values=root_mask, name=f'mask_root')
        ego_nodes = reduce(set.__or__, (set(ego.nodes) for ego in get_wl_label_subgraphs(graph, args.label0)))
        ego_mask = {node: node in ego_nodes for node in graph.nodes}
        set_node_attributes(graph, values=ego_mask, name=f'mask')

if args.label1 is not None:
    for graph in graphs1:
        root_nodes = set(node for node, _ in get_wl_label_nodes(graph, args.label1))
        root_mask = {node: node in root_nodes for node in graph.nodes}
        set_node_attributes(graph, values=root_mask, name=f'mask_root')
        ego_nodes = reduce(set.__or__, (set(ego.nodes) for ego in get_wl_label_subgraphs(graph, args.label1)))
        ego_mask = {node: node in ego_nodes for node in graph.nodes}
        set_node_attributes(graph, values=ego_mask, name=f'mask')


# Save dataset

if args.format == 'networkx':
    with open(os.path.join(args.out, f'{args.name}.pkl'), 'wb') as f:
        pickle.dump({'class0': graphs0, 'class1': graphs1}, f)
elif args.format == 'json':
    for c, graphs in enumerate([graphs0, graphs1]):
        path = os.path.join(args.out, args.name, f'class{c}')
        os.makedirs(path, exist_ok=True)
        for idx, graph in enumerate(graphs):
            with open(os.path.join(path, f'{idx:04d}.json'), 'w') as f:
                json.dump(node_link_data(graph), f)
