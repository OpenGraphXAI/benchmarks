# OpenGraphXAI benchmarks

To run the code in this repository, an Python environment with Pytorch Geometric is required. The instructions to set it up are available [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Loading the OGX benchmarks

We provide a PyG dataset loader in the script `ogx_datasets_pyg.py`.

```python
from ogx_datasets_pyg import OGXBenchmark


dataset = OGXBenchmark(root='/tmp', data_name='alfa')
```

The `mask` attribute in graphs provide the ground truth explanation mask.

## Generating XAI benchmarks

In the directory `src` we provide the scripts to mine candidate benchmarks and to generate benchmarks with chosen Weisfeiler-Leman labels as ground truth.

To mine potential benchmarks from a dataset from the [TU collection](www.graphlearning.io), run

```shell
python make-benchmark-candidates.py --dataset NCI1 --wl 5 --max-labels 100 --min-samples 250 --balance-threshold .8
```

- `--dataset <name>`  selects the original graph classification task to mine

- `--wl <iterations>`  sets the maximum number of WL iterations

- `--max-labels <K>`  is the maximum number of candidate labels to consider for each class

- `--min-samples <N>`  filters out benchmarks that do not meet this minimum number of samples per class

- `--balance-threshold <R>`  filters out benchmarks with minority/majority class imbalance ratio below the specified threshold

To generate a XAI benchmark from the chosen WL labels, run

```shell
python make-benchmark-dataset.py --name alfa --dataset NCI1 --wl 3 --label0 03:d145c98eeabbd3c158ba5aa0ca8b0c2a --label1 03:bc024b8c72e9fb75cd2d8489a726661d --format networkx
```

- `--name <new-name>`  the name of the generated XAI dataset

- `--dataset <name>`  the original graph classification task

- `--label0 <L0>` , `--label1 <L1>`  set the WL labels chosen as ground truth for the respective classes (at least one of them must be specified)

- `--format [networkx|json]` specifies the output format for the new benchmark: either a pickle with a dictionary of networkx graphs, or a plain format where each graph is stored in a JSON file that can be converted in a networkx graph via [`nx.node_link_graph()`](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_graph.html).

In `ogx_benchmarks.csv`  we provide the specifics to generate the 15 benchmarks of the OGX collection, while specifications to generate 2000+ more XAI tasks are provided in `ogx_extended_collection.csv`.


