import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
import numpy as np

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax

class AttentionGNN(torch.nn.Module):
	def __init__(
			self,
			attention_type = 'self',
			propagation = 'label_embed',
			in_dim = 8,
			hid_dim = 64,
			out_dim = 112,
			num_layers = 3,
			dropout = 0.25,
		):
		super(AttentionGNN, self).__init__()

		# create a parameter dictionary to store information about the model, used for logging experiments
		self.param_dict = {'model_type':'ATTN_' + attention_type, 'propagation':propagation, 'in_dim':in_dim, 'hid_dim':hid_dim, 'out_dim':out_dim, 'layers':num_layers,
							'dropout':dropout}

		# construct layers	
		attn_layer = LabelEmbeddingAttentionLayer #LabelFeatureAttentionLayer
		
		self.layers = torch.nn.ModuleList()
		self.layers.append(
			attn_layer(in_dim=in_dim, out_dim=hid_dim)
		)

		for _ in range(num_layers - 2):
			self.layers.append(
				attn_layer(in_dim=hid_dim, out_dim=hid_dim)
			)

		self.layers.append(
			attn_layer(in_dim=hid_dim, out_dim=out_dim)
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


class FeatureAttentionLayer(MessagePassing):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		attention_type = 'self',
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
		self.label_dim = label_dim
		self.label_emb_dim = label_emb_dim

		self.lin_query = Linear(in_dim, attn_heads * out_dim)
		self.lin_key_edge = Linear(edge_dim, attn_heads * out_dim)
		self.lin_key_node = Linear(in_dim, attn_heads * out_dim)
		self.lin_value = Linear(in_dim, attn_heads * out_dim)
		self.lin_skip = Linear(in_dim, attn_heads * out_dim, bias=True)

		self.reset_parameters()

	def reset_parameters(self):
		self.lin_query.reset_parameters()
		self.lin_key_edge.reset_parameters()
		self.lin_key_node.reset_parameters()
		self.lin_value.reset_parameters()
		self.lin_skip.reset_parameters()


	def forward(self, batch):
		H, C = self.heads, self.out_dim

		feat_q = self.lin_query(batch.x).view(-1, H, C)
		feat_v = self.lin_value(batch.x).view(-1, H, C)
		feat_k = self.lin_key_node(batch.x).view(-1, H, C)
	
		# propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
		x = self.propagate(
				batch.edge_index,
				feat_q=feat_q,
				feat_v=feat_v,
				feat_k=feat_k,
				edge_attr=batch.edge_attr,
				size=None,
			)

		x = x.view(-1, self.heads * self.out_dim)

		x_skip = self.lin_skip(batch.x)

		x += x_skip

		return x

	def message(
			self,
			feat_q_i: Tensor,
			feat_v_j: Tensor,
			feat_k_j: Tensor,
			edge_attr: Tensor,
			index: Tensor,
			ptr: OptTensor,
			size_i: Optional[int],
			) -> Tensor:

		edge_k = self.lin_key_edge(edge_attr).view(-1, self.heads, self.out_dim)

		m = self.feature_attention(q=feat_q_i, k=feat_k_j, v=feat_v_j, e=edge_k, index=index)

		return m

	def feature_attention(self, q, k, v, e, index):
		# calculate raw attention scores
		x = q * (e + k)
		x = x.sum(dim=-1)
		x = x / math.sqrt(self.out_dim)

		# normalise attention scores using softmax
		alpha = softmax(x, index)
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		# apply weighted score to neighbour values
		x = v * alpha.view(-1, self.heads, 1)

		return x


class LabelInjectionAttentionLayer(MessagePassing):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		attention_type = 'self',
		propagation = 'feature',
		attn_heads: int = 1,
		dropout: float = 0.1,
		edge_dim: int = 8,
		label_dim: int = 112,
		label_k_dim: int = 8,
		**kwargs,
	):
		kwargs.setdefault('aggr', 'add')
		super(LabelInjectionAttentionLayer, self).__init__(node_dim=0, **kwargs)

		self.in_dim = in_dim
		self.out_dim = out_dim
		self.heads = attn_heads
		self.dropout = dropout
		self.edge_dim = edge_dim
		self.label_dim = label_dim

		# feature layers
		self.lin_query = Linear(in_dim, attn_heads * out_dim)
		self.lin_key_edge = Linear(edge_dim, attn_heads * out_dim)
		self.lin_key_node = Linear(in_dim, attn_heads * out_dim)
		self.lin_value = Linear(in_dim, attn_heads * out_dim)
		self.lin_skip = Linear(in_dim, attn_heads * out_dim, bias=True)

		# label layers
		self.lin_label = Linear(self.label_dim, in_dim, bias=False)

		self.reset_parameters()

	def reset_parameters(self):
#		self.lin_key.reset_parameters()
		self.lin_query.reset_parameters()
		self.lin_value.reset_parameters()
		self.lin_skip.reset_parameters()


	def forward(self, batch):
		H, C = self.heads, self.out_dim

		x = batch.x

		if self.training:
			known_y = batch.train_masked_y	
		else:
			known_y = batch.eval_masked_y
		
		#label = self.lin_label(known_y)

		#x = x + label
		
		feat_q = self.lin_query(batch.x).view(-1, H, C)
		feat_v = self.lin_value(batch.x).view(-1, H, C)
		feat_k = self.lin_key_node(batch.x).view(-1, H, C)
	
		# propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
		x = self.propagate(
				batch.edge_index,
				feat_q=feat_q,
				feat_v=feat_v,
				feat_k=feat_k,
				edge_attr=batch.edge_attr,
				size=None,
			)

		x = x.view(-1, self.heads * self.out_dim)

		x_skip = self.lin_skip(batch.x)

		x += x_skip

		return x

	def message(
			self,
			feat_q_i: Tensor,
			feat_v_j: Tensor,
			feat_k_j: Tensor,
			edge_attr: Tensor,
			index: Tensor,
			ptr: OptTensor,
			size_i: Optional[int],
			) -> Tensor:

		edge_k = self.lin_key_edge(edge_attr).view(-1, self.heads, self.out_dim)

		m = self.self_attention(q=feat_q_i, k=feat_k_j, v=feat_v_j, e=edge_k, index=index)

		return m

	def self_attention(self, q, k, v, e, index):
		# calculate raw attention scores
		x = q * (e + k)
		x = x.sum(dim=-1)
		x = x / math.sqrt(self.out_dim)

		# normalise attention scores using softmax
		alpha = softmax(x, index)
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		# apply weighted score to neighbour values
		x = v * alpha.view(-1, self.heads, 1)

		return x

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}({self.in_channels}, '
				f'{self.out_channels}, heads={self.heads})')


class LabelEmbeddingAttentionLayer(MessagePassing):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		attention_type = 'self',
		propagation = 'both',
		attn_heads: int = 1,
		dropout: float = 0.1,
		edge_dim: int = 8,
		label_dim: int = 112,
		label_emb_dim: int = 8,
		label_k: int = 4,
		**kwargs,
	):
		kwargs.setdefault('aggr', 'add')
		super(LabelEmbeddingAttentionLayer, self).__init__(node_dim=0, **kwargs)

		self.in_dim = in_dim
		self.out_dim = out_dim
		self.heads = attn_heads
		self.dropout = dropout
		self.edge_dim = edge_dim
		self.label_dim = label_dim
		self.label_k = label_k

		# feature layers
		self.lin_query = Linear(in_dim, attn_heads * out_dim)
		self.lin_key_edge = Linear(edge_dim, attn_heads * out_dim)
		self.lin_key_node = Linear(in_dim, attn_heads * out_dim)
		self.lin_value = Linear(in_dim, attn_heads * out_dim)
		self.lin_skip = Linear(in_dim, attn_heads * out_dim, bias=True)

		# label layers
		self.lin_label = Linear(self.label_dim, in_dim, bias=False)
		self.emb_label = torch.nn.Parameter(torch.randn(label_dim, out_dim))
		self.emb_label.required_grad = True
		torch.nn.init.xavier_uniform(self.emb_label)
		self.lin_label_to_k = Linear(self.label_dim, label_k)
		self.lin_k_to_out = Linear(label_k, out_dim)
		self.label_softmax = torch.nn.Softmax(dim=1)

		self.lin_comb = Linear(out_dim*3, out_dim)

		self.reset_parameters()

	def reset_parameters(self):
#		self.lin_key.reset_parameters()
		self.lin_query.reset_parameters()
		self.lin_value.reset_parameters()
		self.lin_skip.reset_parameters()


	def forward(self, batch):
		H, C = self.heads, self.out_dim

		x = batch.x

		if self.training:
			mask = batch.train_mask
		else:
			mask = batch.eval_mask
		
		#label = self.lin_label(known_y)

		#x = x + label
		
		feat_q = self.lin_query(batch.x).view(-1, H, C)
		feat_v = self.lin_value(batch.x).view(-1, H, C)
		feat_k = self.lin_key_node(batch.x).view(-1, H, C)
	
		# propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
		m = self.propagate(
				batch.edge_index,
				feat_q=feat_q,
				feat_v=feat_v,
				feat_k=feat_k,
				edge_attr=batch.edge_attr,
				label = batch.y,
				mask = mask,
				size=None,
			)

		m = m.squeeze()
		x_skip = self.lin_skip(batch.x)

		#print(batch.x.shape, m.shape)

		x = torch.cat([x_skip, m], dim=-1)
		
		x = self.lin_comb(x)

		return x

	def message(
			self,
			feat_q_i: Tensor,
			feat_v_j: Tensor,
			feat_k_j: Tensor,
			edge_attr: Tensor,
			label_j: Tensor,
			mask_j: Tensor,
			index: Tensor,
			ptr: OptTensor,
			size_i: Optional[int],
			) -> Tensor:

		edge_k = self.lin_key_edge(edge_attr).view(-1, self.heads, self.out_dim)

		#f = self.feature_attention(q=feat_q_i, k=feat_k_j, v=feat_v_j, e=edge_k, index=index)
		l = self.label_attention(q=feat_q_i, l=label_j, e=edge_k, mask=mask_j, index=index) 
		
		m = torch.cat([f,l], dim=-1)

		return m

	def feature_attention(self, q, k, v, e, index):
		# calculate raw attention scores
		x = q * (e + k)
		x = x.sum(dim=-1)
		x = x / math.sqrt(self.out_dim)

		# normalise attention scores using softmax
		alpha = softmax(x, index)
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		# apply weighted score to neighbour values
		x = v * alpha.view(-1, self.heads, 1)
		return x

	def label_attention(self, q, l, e, mask, index):
		# multiply label by embedding matrix (acts as a mask)
		embedded_labels = l.unsqueeze(-1) * self.emb_label
		embedded_labels = embedded_labels

		# project sequence dimension to k
		k_labels = self.lin_label_to_k(embedded_labels.permute(0,2,1)).permute(0,2,1)

		# self-attention
		q = q.repeat(1, self.label_k, 1)
		x = q * (e + k_labels)
		x = x.sum(dim=-1)
		x = x / math.sqrt(self.label_k)
		x = x + (mask * -np.inf)
		print(x)
		exit()

		alpha = self.label_softmax(x)
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		x = x * alpha
		x = self.lin_k_to_out(x).unsqueeze(1) 
		
		return x	
		

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}({self.in_channels}, '
				f'{self.out_channels}, heads={self.heads})')
