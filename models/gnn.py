import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv

class GNN(torch.nn.Module):
	'''
	General class for creating different kinds of Graph Neural Networks.
	paramets:
		- conv_type = the type of convolutional layer to use, e.g. 'GCN', 'SAGE', 'GAT'
		- in_dim = the dimensionality of the input
		- hid_dim = the dimensionality of the hidden dimensions of the network
		- out_dim = the dimensionality of the output
		- num_layers = the number of hidden layers to use between the input and output layers
		- dropout = the dropout probability to use
	'''
	def __init__(
			self,
			conv_type = 'GCN',
			in_dim = 8,
			hid_dim = 64,
			out_dim = 112,
			num_layers = 3,
			dropout = 0.25,
			):
		super(GNN, self).__init__()

		# create a parameter dictionary to store information about the model, used for logging experiments
		self.param_dict = {'model_type':'GNN_' + conv_type, 'in_dim':in_dim, 'hid_dim':hid_dim, 'out_dim':out_dim, 'layers':num_layers,
							'dropout':dropout}
		
		# set the convolutional layer type to use in the model
		if conv_type == 'GCN':
			layer = GCNConv
		elif conv_type == 'SAGE':
			layer = SAGEConv
		elif conv_type == 'GAT':
			layer = GATConv
		elif conv_type == 'TransformerConv':
			layer = TransformerConv
		else:
			raise Exception('GNN model type "' + conv_type + '" not recognized')

		# initialise network layers
		self.layers = torch.nn.ModuleList()
		self.layers.append(
			layer(in_dim, hid_dim)
			)
	
		for _ in range(num_layers - 2):
			self.layers.append(
				layer(hid_dim, hid_dim))
	
		self.layers.append(
			layer(hid_dim, out_dim))

		self.dropout = dropout

	def reset_parameters(self):
		for layer in self.layers:
			layer.reset_parameters()

	def forward(self, batch):
		for layer in self.layers[:-1]:
			batch.x = layer(batch.x, batch.edge_index)
			batch.x = F.relu(batch.x)
			batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
		batch.x = self.layers[-1](batch.x, batch.edge_index)
		return batch.x


