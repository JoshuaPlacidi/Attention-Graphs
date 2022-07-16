import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax

class AttentionGNN(torch.nn.Module):
	def __init__(
			self,
			attention_type = 'self',
			in_dim = 8,
			hid_dim = 64,
			out_dim = 112,
			num_layers = 3,
			dropout = 0.25,
		):
		super(AttentionGNN, self).__init__()

		# create a parameter dictionary to store information about the model, used for logging experiments
		self.param_dict = {'model_type':'ATTN_' + attention_type, 'in_dim':in_dim, 'hid_dim':hid_dim, 'out_dim':out_dim, 'layers':num_layers,
							'dropout':dropout}
	
		attn_layer = AttentionLayer
		
		self.layers = torch.nn.ModuleList()
		self.layers.append(
			attn_layer(in_dim, hid_dim)
		)

		for _ in range(num_layers - 2):
			self.layers.append(
				attn_layer(hid_dim, hid_dim)
			)

		self.layers.append(
			attn_layer(hid_dim, out_dim)
		)

		self.dropout = dropout

	def reset_parameters(self):
		for layer in self.layers:
			layer.reset_parameters()

	def forward(self, batch):
		for layer in self.layers[:-1]:
			batch.x = layer(batch)
			batch.x = F.relu(batch.x)
			batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
		batch.x = self.layers[-1](batch)
		return batch.x


class AttentionLayer(MessagePassing):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		attn_heads: int = 1,
		dropout: float = 0.1,
		edge_dim: int = 8,
		**kwargs,
	):
		kwargs.setdefault('aggr', 'add')
		super(AttentionLayer, self).__init__(node_dim=0, **kwargs)

		self.in_dim = in_dim
		self.out_dim = out_dim
		self.heads = attn_heads
		self.dropout = dropout
		self.edge_dim = edge_dim

		self.lin_query = Linear(in_dim, attn_heads * out_dim)
		self.lin_key = Linear(edge_dim, attn_heads * out_dim)
		self.lin_value = Linear(in_dim, attn_heads * out_dim)
		self.lin_skip = Linear(in_dim, attn_heads * out_dim, bias=True)

		self.reset_parameters()

	def reset_parameters(self):
		self.lin_key.reset_parameters()
		self.lin_query.reset_parameters()
		self.lin_value.reset_parameters()
		self.lin_skip.reset_parameters()


	def forward(self, batch):
		H, C = self.heads, self.out_dim

		query = self.lin_query(batch.x).view(-1, H, C)
		value = self.lin_value(batch.x).view(-1, H, C)
	
		# propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
		x = self.propagate(batch.edge_index, query=query, edge_attr=batch.edge_attr, value=value, size=None)

		x = x.view(-1, self.heads * self.out_dim)

		x_skip = self.lin_skip(batch.x)

		x += x_skip

		return x

	def message(self, query_i: Tensor, edge_attr: Tensor, value_j: Tensor,
				index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:

		key = self.lin_key(edge_attr).view(-1, self.heads, self.out_dim)
		m = self.self_attention(query=query_i, key=key, value=value_j, index=index)

		return m

	def self_attention(self, query, key, value, index):
		x = query * key
		x = x.sum(dim=-1)
		x = x / math.sqrt(self.out_dim)
		
		alpha = softmax(x, index)
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		x = value * alpha.view(-1, self.heads, 1)
		return x

	def mlp_attention(self, node_i, node_j, edge, label_j):
		raise NotImplemented
		# x = torch.concat([node_i, node_j, edge, label_j], dim=-1)
		# x = 

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}({self.in_channels}, '
				f'{self.out_channels}, heads={self.heads})')
