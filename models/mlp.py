import torch
import torch.nn as nn
import math
from collections import OrderedDict

class mlp(nn.Module):
	def __init__(self, input_size, hidden_size, out_size,
				 data_src=None, num_layers=1, dropout_p=0.0):
		"""Constructor for MLP.
		Args:
			input_size: The number of input dimensions.
			hidden_size: The number of hidden dimensions for each layer.
			num_classes: The size of the output.
			num_layers: The number of hidden layers.
			dropout_p: Dropout probability.
		"""
		super(mlp, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.out_size = out_size

		layers = []
		for i in range(num_layers):
			idim = hidden_size
			odim = hidden_size
			if i == 0:
				idim = input_size
			if i == num_layers-1:
				odim = out_size
			fc = nn.Linear(idim, odim)
			fc.weight.data.normal_(0.0,  math.sqrt(2. / idim))
			fc.bias.data.fill_(0)
			layers.append(('fc'+str(i), fc))
			if i != num_layers-1:
				layers.append(('relu'+str(i), nn.ReLU()))
				layers.append(('dropout'+str(i), nn.Dropout(p=dropout_p)))
		
		self.layers = nn.Sequential(OrderedDict(layers))

	def params_to_train(self):
		return self.layers.parameters()

	def forward(self, e, l):
		e = torch.sum(e, dim=1)
		l = torch.sum(l, dim=1)

		out = self.layers(e)
		return out
