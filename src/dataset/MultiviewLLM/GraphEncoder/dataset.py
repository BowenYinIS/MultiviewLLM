import torch
import numpy as np


def make_dataset(dataset_path):
    '''
    Load the graph dataset from the specified path.

    Args:
        dataset_path (str or Path): Path to the saved dataset file.

    Returns:
        list: List of graph data objects.
    '''
    dataset = torch.load(dataset_path, weights_only=False)

    for key, value in dataset.items():
        print('Split:', key)
        print(f"# of graphs: {len(value)}, \n"
              f"# of nodes: {np.mean([data.num_nodes for data in value])}, \n"
              f"# of edges: {np.mean([data.num_edges for data in value])}, \n"
              f"# of features: {value[0].num_features}, \n"
              )
        print('---'*10)

    dataset = dataset['train'] + dataset['test']
    return dataset
