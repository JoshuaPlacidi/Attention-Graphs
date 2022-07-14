import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv
from models.transformers import TransformerConvLayer
from models.mlp import MLP_layer

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
	def __init__(self,
			conv_type = 'GCN',
			in_dim = 8,
			hid_dim = 256,
			out_dim = 112,
			num_layers = 3,
			dropout = 0.25,
			):
		super(GNN, self).__init__()

		# create a parameter dictionary to store information about the model, used for logging experiments
		self.param_dict = {'model_type':conv_type, 'in_dim':in_dim, 'hid_dim':hid_dim, 'out_dim':out_dim, 'layers':num_layers,
							'dropout':dropout}
		
		# set the convolutional layer type to use in the model
		if conv_type == 'GCN':
			conv_layer = GCNConv
		elif conv_type == 'SAGE':
			conv_layer = SAGEConv
		elif conv_type == 'GAT':
			conv_layer = GATConv
		elif conv_type == 'TransformerConv':
			conv_layer = TransformerConvLayer#
		else:
			raise Exception('GNN model type "' + conv_type + '" not recognized')

#		conv_layer = MLP_layer

		# initialise network layers
		self.convs = torch.nn.ModuleList()
		self.convs.append(
			conv_layer(in_dim, hid_dim))
	
		for _ in range(num_layers - 2):
			self.convs.append(
				conv_layer(hid_dim, hid_dim))
	
		self.convs.append(
			conv_layer(hid_dim, out_dim))

		self.dropout = dropout

	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()

	def forward(self, batch):
		x, adj_t = batch.x, batch.edge_index
		for conv in self.convs[:-1]:
			batch.x = conv(batch) #x = conv(x, adj_t)
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)
		batch.x = self.convs[-1](batch)#(x, adj_t)
		return batch.x


