import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
import math
from collections import OrderedDict

class MLP(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim, num_layers=1, dropout=0.0):
		'''
		Multi-Layer Perceptron
		params:
			- in_dim: input dimension
			- hid_dim: hidden dimension
			- out_dim: output dimension
			- num_layers: number of HIDDEN layers, additional input and output layers are added
			- dropout: the probability of dropping out an input
		'''
		super(MLP, self).__init__()
		self.param_dict = {'model_type':'MLP', 'in_dim':in_dim, 'hid_dim':hid_dim, 'out_dim':out_dim, 'layers':num_layers,
							'dropout':dropout}

		layers = []
		for i in range(num_layers):
			idim = hid_dim
			odim = hid_dim
			if i == 0:
				idim = in_dim
			if i == num_layers-1:
				odim = out_dim
			fc = nn.Linear(idim, odim)
			fc.weight.data.normal_(0.0,  math.sqrt(2. / idim))
			fc.bias.data.fill_(0)
			layers.append(('fc'+str(i), fc))
			if i != num_layers-1:
				layers.append(('relu'+str(i), nn.ReLU()))
				layers.append(('dropout'+str(i), nn.Dropout(p=dropout)))
		
		self.layers = nn.Sequential(OrderedDict(layers))
		self.sigmoid = nn.Sigmoid()

	def reset_parameters(self):
		for layer in self.layers:
			if isinstance(layer, nn.Linear):
				layer.reset_parameters()

	def params_to_train(self):
		return self.layers.parameters()

	def forward(self, batch):
		x = batch.x[:batch.batch_size]
		x = self.layers(x)
		return x
