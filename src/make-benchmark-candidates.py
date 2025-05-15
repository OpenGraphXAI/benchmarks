import argparse
import itertools
from collections import Counter

import pandas as pd
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from graph_histogram import weisfeiler_lehman_graph_histogram


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name from TU collection', required=True)
parser.add_argument('--root', help='root directory for source dataset', default='/tmp')
parser.add_argument('--wl', help='WL coloring iterations', type=int, default=5)
parser.add_argument('--hash-size', help='has digest size for WL labels', type=int, default=16)
parser.add_argument('--max-labels', help='maximum number of top WL labels per class', type=int, default=10)
parser.add_argument('--min-samples', help='minimum number of samples per class', type=int, default=10)
parser.add_argument('--balance-threshold', help='minimum threshold for minority/majority samples ratio', type=float, default=0)
parser.add_argument('--out', help='output csv file of benchmark candidates', default=None)
args = parser.parse_args()


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


# Count WL label occurrences and prevalences in classes

label_counts0 = sum((Counter({k: 1 for k in histogram}) for histogram, y in zip(all_histograms, targets) if y == 0), start=Counter())
label_counts1 = sum((Counter({k: 1 for k in histogram}) for histogram, y in zip(all_histograms, targets) if y == 1), start=Counter())

label_classes = {k: [0, 0] for k in itertools.chain(label_counts0, label_counts1)}
for label, count in label_counts0.items():
    label_classes[label][0] += count
for label, count in label_counts1.items():
    label_classes[label][1] += count

label_diff = {k: count[1] - count[0] for k, count in label_classes.items()}


# Generate all candidate benchmarks

top_label0 = list(itertools.islice((key for key, _ in sorted(label_diff.items(), key=lambda item: item[1])), args.max_labels))
top_label1 = list(itertools.islice((key for key, _ in sorted(label_diff.items(), key=lambda item: item[1], reverse=True)), args.max_labels))

tasks = []

# Discriminating labels for each class
for label0, label1 in itertools.product(top_label0, top_label1):
    count0 = sum(1 for histogram, y in zip(all_histograms, targets) if label0 in histogram and label1 not in histogram and y == 0)
    count1 = sum(1 for histogram, y in zip(all_histograms, targets) if label0 not in histogram and label1 in histogram and y == 1)
    tasks.append({'label0': label0, 'label1': label1, 'count0': count0, 'count1': count1})

# Discriminating label only for class 0
for label0 in top_label0:
    count0 = sum(1 for histogram, y in zip(all_histograms, targets) if label0 in histogram and y == 0)
    count1 = sum(1 for histogram, y in zip(all_histograms, targets) if label0 not in histogram and y == 1)
    tasks.append({'label0': label0, 'label1': None, 'count0': count0, 'count1': count1})

# Discriminating label only for class 1
for label1 in top_label1:
    count0 = sum(1 for histogram, y in zip(all_histograms, targets) if label1 not in histogram and y == 0)
    count1 = sum(1 for histogram, y in zip(all_histograms, targets) if label1 in histogram and y == 1)
    tasks.append({'label0': None, 'label1': label1, 'count0': count0, 'count1': count1})


# Filter and save candidates

tasks = [item | {'dataset': dataset.name, 'samples': item['count0'] + item['count1'], 'balance': min(item['count0'], item['count1']) / max(item['count0'], item['count1'])} for item in tasks if min(item['count0'], item['count1']) > 0]

df = pd.DataFrame(sorted(tasks, key=lambda item: min(item['count0'], item['count1']), reverse=True), columns=['dataset', 'label0', 'label1', 'count0', 'count1', 'samples', 'balance'])

df[df.count0 >= args.min_samples][df.count1 >= args.min_samples][df.balance >= args.balance_threshold].to_csv(args.out or f'{args.dataset}.XAI.csv', index=False, float_format='%.2f')
