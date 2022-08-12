import torch
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
import config

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv

class GNN(torch.nn.Module):
	def __init__(
			self,
			conv_type = 'GCN',
			propagation = 'feature',
			label_process = 'linear',
			in_dim = 8,
			hid_dim = 64,
			out_dim = 112,
			num_layers = 3,
			dropout = 0.25,
			):
		'''
		General class for creating different kinds of Graph Neural Networks.
		paramets:
			- conv_type: the type of convolutional layer to use, e.g. 'GCN', 'SAGE', 'GAT'
			- propagation: the source of information to propagate, can be 'feature', 'label', or 'both'
			- label_process: how label information is preprocessed can be 'linear' or 'embed'
			- in_dim: the dimensionality of the input
			- hid_dim: the dimensionality of the hidden dimensions of the network
			- out_dim: the dimensionality of the output
			- num_layers: the number of hidden layers to use between the input and output layers
			- dropout: the dropout probability to use
		'''

		super(GNN, self).__init__()

		# create a parameter dictionary to store information about the model, used for logging experiments
		self.param_dict = {'model_type':'GNN_' + conv_type, 'propagation':propagation, 'in_dim':in_dim, 'hid_dim':hid_dim, 'out_dim':out_dim, 'layers':num_layers,
							'dropout':dropout, 'label_process':label_process}
		
		self.propagation = propagation
		self.label_process = label_process
	
		# set the convolutional layer type to use in the model
		if conv_type == 'GCN':
			layer = GCNConv
		elif conv_type == 'SAGE':
			layer = SAGEConv
		elif conv_type == 'GAT':
			layer = GATConv
		elif conv_type == 'TFC':
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

		if propagation != 'feature':
			
			if label_process == 'embed':
				# label embedding table
				self.label_embedding_table = torch.nn.Parameter(torch.randn(out_dim, 8))
				self.label_embedding_table.required_grad = True

			elif label_process == 'linear':
				self.lin_label = Linear(112, 8, bias=True)

			elif label_process == 'atn_embed':
				self.label_embedding_table = torch.nn.Parameter(torch.randn(out_dim, 8))
				self.label_embedding_table.required_grad = True

				self.lin_feat_query = Linear(in_dim, out_dim)
				self.lin_label_key = Linear(112, out_dim)
				self.lin_label_value = Linear(112, out_dim)
				

	def reset_parameters(self):
		for layer in self.layers:
			layer.reset_parameters()

		if self.propagation != 'feature':
			if self.label_process == 'linear':
				self.lin_label.reset_parameters()

			elif self.label_process == 'embed' or self.label_process == 'atn_embed':
				self.label_embedding_table = torch.nn.init.xavier_uniform_(self.label_embedding_table)

	def get_input(self, batch):
		if self.propagation == 'feature':
			return batch.features

		else:
			if self.training:
				mask = batch.train_mask
			else:
				mask = batch.eval_mask

			if self.label_process == 'linear':
				labels = self.lin_label(batch.y.float().to(config.device))

			elif self.label_process == 'embed':
				labels = batch.y.unsqueeze(-1).to(config.device) * self.label_embedding_table
				labels = labels.sum(dim=1)

			elif self.label_process == 'embed_atn':
				labels = batch.y.unsqueeze(-1).to(config.device) * self.label_embedding_table


			inverted_mask = torch.ones_like(mask) - mask
			labels = labels * inverted_mask.to(config.device)

			if self.propagation == 'label':
				return labels
			elif self.propagation == 'both':
				return torch.cat([batch.features.to(config.device),labels], dim=1)

		raise Exception('Input processing request failed')

	def forward(self, batch):
		# initialise x depending on the propagation type
		x = self.get_input(batch).to(config.device)
		edge_index = batch.edge_index.to(config.device)

		# for each layer of the network
		for layer in self.layers[:-1]:

			#pass input and apply activation and dropout
			x = layer(x, edge_index)
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)

		# project to output dimension
		x = self.layers[-1](x, edge_index)

		return x


