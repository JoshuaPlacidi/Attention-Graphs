import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

class GCN(torch.nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim, num_layers,
				 dropout):
		super(GCN, self).__init__()

		self.param_dict = {'in_dim':in_dim, 'hid_dim':hid_dim, 'out_dim':out_dim, 'layers':num_layers,
							'dropout':dropout}

		self.convs = torch.nn.ModuleList()
		self.convs.append(
			GCNConv(in_dim, hid_dim, normalize=False))
		for _ in range(num_layers - 2):
			self.convs.append(
				GCNConv(hid_dim, hid_dim, normalize=False))
		self.convs.append(
			GCNConv(hid_dim, out_dim, normalize=False))

		self.dropout = dropout

	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()

	def forward(self, x, adj_t):
		for conv in self.convs[:-1]:
			x = conv(x, adj_t)
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.convs[-1](x, adj_t)
		return x

class SAGE(torch.nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim, num_layers,
				 dropout):
		super(SAGE, self).__init__()

		self.param_dict = {'in_dim':in_dim, 'hid_dim':hid_dim, 'out_dim':out_dim, 'layers':num_layers,
							'dropout':dropout}

		self.convs = torch.nn.ModuleList()
		self.convs.append(SAGEConv(in_dim, hid_dim))
		for _ in range(num_layers - 2):
			self.convs.append(SAGEConv(hid_dim, hid_dim))
		self.convs.append(SAGEConv(hid_dim, out_dim))

		self.dropout = dropout

	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()

	def forward(self, x, adj_t):
		for conv in self.convs[:-1]:
			x = conv(x, adj_t)
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.convs[-1](x, adj_t)
		return x 
