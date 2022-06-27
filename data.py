import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from torch.utils.data import Dataset, DataLoader

class tensor_dataset(Dataset):
	def __init__(self, graph, indices):	
		self.adj_row, self.adj_col, self.adj_val = graph.adj_t.coo()
		self.y = graph.y
		self.indices = indices

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		# get index of datapoint in graph tensors
		i = self.indices[idx]

		# get the edge indices for datapoint i
		edge_indices = (self.adj_row==i).nonzero().squeeze()

		# get features for edges
		edge_features = torch.index_select(self.adj_val, 0, edge_indices)
		
		# get indices of neighbouring nodes
		neighbour_i = torch.index_select(self.adj_col, 0, edge_indices)
		# get labels of neighbours
		neighbour_labels = torch.index_select(self.y, 0, neighbour_i)

		# pad tensors
		pad_to = 7750
		pad = (0, 0, 0, pad_to-edge_features.shape[0])
		edge_features = torch.nn.functional.pad(edge_features, pad, "constant", 0)
		neighbour_labels = torch.nn.functional.pad(neighbour_labels, pad, "constant", 0)	
		
		return edge_features, neighbour_labels, self.y[i]

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


def get_graph():
	dataset = PygNodePropPredDataset(name='ogbn-proteins', root="/Users/joshua/env/datasets")#, transform=T.ToSparseTensor(attr='edge_attr'))
	split_idx = dataset.get_idx_split()

	graph = dataset[0]
#	graph.x = graph.adj_t.mean(dim=1)
#	graph.adj_t.set_value_(None)	

	return graph, split_idx


def get_proteins_data_dicts(batch_size=32, return_graph_dataset=False):
	dataset = PygNodePropPredDataset(name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'), root="/Users/joshua/env/datasets")
	graph = dataset[0]

	split_idx = dataset.get_idx_split()
	data_dict = {}

	sample_sets = ['train', 'valid', 'test']
	
	if return_graph_dataset:
		return get_graph_data(graph), split_idx

	else:

		for s in sample_sets:
			data_dict[s] = DataLoader(
								dataset(graph, indices=split_idx[s]),
								batch_size=batch_size,
								shuffle=True,
								num_workers=0
							)
	
		return data_dict
