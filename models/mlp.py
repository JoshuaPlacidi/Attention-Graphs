import torch
import torch.nn as nn
import math
from collections import OrderedDict

class MLP(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim,
				 data_src=None, num_layers=1, dropout=0.0):
		"""Constructor for MLP.
		Args:
			input_size: The number of input dimensions.
			hidden_size: The number of hidden dimensions for each layer.
			num_classes: The size of the output.
			num_layers: The number of hidden layers.
			dropout_p: Dropout probability.
		"""
		super(MLP, self).__init__()
		self.param_dict = {'in_dim':in_dim, 'hid_dim':hid_dim,
							'out_dim': out_dim, 'dropout':dropout}

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
		x = self.layers(batch.x)
		return x
