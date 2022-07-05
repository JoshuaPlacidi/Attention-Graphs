import torch
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

class graph_dataset(Dataset):
	def __init__(self, graph, indicies):
		self.edges = graph.adj_t
		self.y = graph.y
		self.indices = indicies

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		i = self.indices[idx]
		return self.edges[i], self.y[i]


def get_graph_data():
	data = PygNodePropPredDataset(name='ogbn-proteins', root="/Users/joshua/env/datasets")
	split_idx = data.get_idx_split()

	graph = data[0]

	return graph, split_idx

