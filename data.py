import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from torch.utils.data import Dataset, DataLoader

class proteins_dataset(Dataset):
	def __init__(self, graph, indices=None):	
		self.adj_row, self.adj_col, self.adj_val = graph.adj_t.coo()

		self.y = graph.y
		self.indices = indices

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		# get index of datapoint in graph tensors
		i = self.indices[idx]

		# get the edge indices for datapoint i
	#	print('run', i.item(), end=' - ')
		edge_indices = (self.adj_row==i).nonzero().squeeze()
	#	print('complete')

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


def get_proteins_data_dict():
	dataset = PygNodePropPredDataset(name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'), root="/Users/joshua/dev/datasets")
	graph = dataset[0]

	split_idx = dataset.get_idx_split()
	train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

	data_dict = {
		'train': DataLoader(proteins_dataset(graph, indices=train_idx), batch_size=32, shuffle=True, num_workers=0),
		'val': DataLoader(proteins_dataset(graph, indices=val_idx), batch_size=32, shuffle=True, num_workers=0),
		'test': DataLoader(proteins_dataset(graph, indices=test_idx), batch_size=32, shuffle=True, num_workers=0)
	}
	return data_dict
