import config
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax, degree

class MultiPropGNN(torch.nn.Module):
	def __init__(
			self,
			label_head = 'gat', # propagation algorithm to use for propagating labels: gcn, gat, sage, tf
			in_dim = 8,
			hid_dim = 64,
			out_dim = 112,
			label_emb_dim = 8,
			num_layers = 3,
			dropout = 0.25,
		):
		super(MultiPropGNN, self).__init__()

		# create a parameter dictionary to store information about the model, used for logging experiments
		self.param_dict = {'model_type':'MultiProp', 'propagation':'both', 'in_dim':in_dim, 'hid_dim':hid_dim, 'out_dim':out_dim, 'layers':num_layers,
							'dropout':dropout}
		self.dropout = dropout

		# label embedding table
		self.label_embedding_table = torch.nn.Parameter(torch.randn(out_dim, label_emb_dim))
		self.label_embedding_table.required_grad = True

		self.label_head = label_head

		# construct layers	
		self.layers = torch.nn.ModuleList()
		self.layers.append(
			MultiPropLayer(label_head=label_head, in_dim=in_dim, out_dim=hid_dim, dropout=dropout, label_embedding_table=self.label_embedding_table)
		)
		for _ in range(num_layers - 2):
			self.layers.append(
				MultiPropLayer(label_head=label_head, in_dim=hid_dim, out_dim=hid_dim, dropout=dropout, label_embedding_table=self.label_embedding_table)
			)
		self.layers.append(
			MultiPropLayer(label_head=label_head, in_dim=hid_dim, out_dim=out_dim, dropout=dropout, label_embedding_table=self.label_embedding_table)
		)

		self.reset_parameters()

	def reset_parameters(self):
		for layer in self.layers:
			layer.reset_parameters()
		
		self.label_embedding_table = torch.nn.init.xavier_uniform(self.label_embedding_table)

	def forward(self, batch):
		for layer in self.layers[:-1]:
			batch.features = layer(batch)
			batch.features = F.relu(batch.features)
			batch.features = F.dropout(batch.features, p=self.dropout, training=self.training)
		batch.features = self.layers[-1](batch)
		return batch.features


class MultiPropLayer(MessagePassing):

	def __init__(
		self,
		label_head = 'tf',
		in_dim: int = 8,
		out_dim: int = 64,
		label_embedding_table = None,
		attn_heads: int = 1,
		dropout: float = 0.25,
		edge_dim: int = 8,
		label_dim: int = 112,
		label_k: int = 16,
		**kwargs,
	):
		kwargs.setdefault('aggr', 'add')
		super(MultiPropLayer, self).__init__(node_dim=0, **kwargs)

		if label_head not in ['tf', 'sage', 'gat', 'gcn']:
			raise Exception("'{0} label head argument not recognized".format(label_head))

		self.label_head = label_head
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.heads = attn_heads
		self.dropout = dropout
		self.edge_dim = edge_dim
		self.label_dim = label_dim
		self.label_k = label_k
		self.label_embedding_table = label_embedding_table

		# feature attention layers
		self.lin_feat_query = Linear(in_dim, attn_heads * out_dim)
		self.lin_feat_key_node = Linear(in_dim, attn_heads * out_dim)
		self.lin_feat_value = Linear(in_dim, attn_heads * out_dim)
		self.lin_feat_skip = Linear(in_dim, attn_heads * out_dim, bias=True)

		# label propagation layers
		if self.label_head == 'tf':
			self.lin_label_key = Linear(out_dim, out_dim)
		elif self.label_head == 'gat':
			self.lin_alpha = Linear(2*out_dim, 1)
		elif self.label_head == 'sage':
			self.lin_message = Linear(out_dim, out_dim)

		# inner label attention
		self.lin_label_emb_to_out = Linear(self.label_embedding_table.shape[1], out_dim)
		self.lin_label_in_to_k = Linear(self.label_dim, label_k)
		self.softmax_label_k = torch.nn.Softmax(dim=1)
		self.lin_label_k_key = Linear(out_dim, out_dim)

		# combination layers
		self.lin_key_edge = Linear(edge_dim, attn_heads * out_dim)
		self.lin_comb = Linear(out_dim*2, out_dim)

	def reset_parameters(self):
		self.lin_feat_query.reset_parameters()
		self.lin_feat_key_node.reset_parameters()
		self.lin_feat_value.reset_parameters()
		self.lin_feat_skip.reset_parameters()

		if self.label_head == 'tf':
			self.lin_label_key.reset_parameters()
		elif self.label_head == 'gat':
			self.lin_alpha.reset_parameters()
		elif self.label_head == 'sage':
			self.lin_message.reset_parameters()

		self.lin_label_in_to_k.reset_parameters()
		self.lin_label_k_key.reset_parameters()

		self.lin_key_edge.reset_parameters()
		self.lin_comb.reset_parameters()


	def forward(self, batch):
		H, C = self.heads, self.out_dim

		batch = batch.to(config.device)

		if self.training:
			mask = batch.train_mask
		else:
			mask = batch.eval_mask

		feat_q = self.lin_feat_query(batch.features).view(-1, H, C)
		feat_v = self.lin_feat_value(batch.features).view(-1, H, C)
		feat_k = self.lin_feat_key_node(batch.features).view(-1, H, C)

		# calculate node messages
		m = self.propagate(
				batch.edge_index,
				feat_q=feat_q,
				feat_v=feat_v,
				feat_k=feat_k,
				edge_attr=batch.edge_attr,
				label = batch.y,
				mask = mask,
				size=None,
			).squeeze()

		if self.label_head == 'sage':
			m = self.lin_message(m)

		feature_skip = self.lin_feat_skip(batch.features)

		out = torch.cat([feature_skip, m], dim=-1)
		
		out = self.lin_comb(out)

		#out = m

		return out

	def message(
			self,
			feat_q_i: Tensor,
			feat_v_j: Tensor,
			feat_k_j: Tensor,
			edge_attr: Tensor,
			label_j: Tensor,
			mask_j: Tensor,
			index: Tensor,
			) -> Tensor:

		edge_k = self.lin_key_edge(edge_attr).view(-1, self.heads, self.out_dim)

		# calculate feature messages
		#feature_messages = self.self_attention(q=feat_q_i, k=feat_k_j, v=feat_v_j, e=edge_k, index=index)

		# calculate label messages
		label_messages = self.label_attention(q=feat_q_i, l=label_j, e=edge_k, mask=mask_j, index=index)
		
		#messages = torch.cat([feature_messages, label_messages], dim=-1)

		return label_messages #messages

	def self_attention(self, q, k, v, e, index, mask=None):
		# calculate raw attention scores
		x = q * (e + k)
		x = x.sum(dim=-1)
		x = x / math.sqrt(self.out_dim)

		# apply mask if it is present
		if mask != None:
			# calculate label mask, if a label is masked we set its alpha to a large negative
			softmax_mask = mask * -1e6
			softmax_mask += torch.ones_like(softmax_mask)
			x = x * softmax_mask

		# normalise attention scores using softmax
		alpha = softmax(x, index)
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		# apply weighted score to neighbour values
		out = v * alpha.view(-1, self.heads, 1)

		# zero out mask outputs to prevent label leakage
		if mask != None:
			# mask out any labels that might have been leaked due to uniform softmax
			inverted_mask = (torch.ones_like(mask) - mask).unsqueeze(-1)
			out = out * inverted_mask

		return out

	def label_attention(self, q, l, e, mask, index):
		# multiply label by embedding matrix (acts as a mask)
		embedded_labels = l.unsqueeze(-1) * self.label_embedding_table

		# project sequence dimension to k
		k_labels = self.lin_label_in_to_k(embedded_labels.permute(0,2,1)).permute(0,2,1)
		k_labels = self.lin_label_emb_to_out(k_labels)
		k_labels_key = self.lin_label_k_key(k_labels)

		# self-attention
		q_k = q.repeat(1, self.label_k, 1)
		x = q_k * (e + k_labels_key)
		x = x.sum(dim=-1)
		x = x / math.sqrt(self.label_k)

		# apply softmax and apply dropout
		alpha = self.softmax_label_k(x)
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		# weighed sum of labels over k dimension into 1 represention
		out = k_labels * alpha.view(-1, self.label_k, 1)
		out = out.sum(dim=1).view(-1, 1, self.out_dim)

		# propagate labels
		if self.label_head == 'tf':
			out_key = self.lin_label_key(out)
			out = self.self_attention(q, out_key, out, e, index, mask=mask)
		elif self.label_head == 'gcn':
			out = self.gcn(x_j=out, e=e, mask=mask)
		elif self.label_head == 'gat':
			out = self.gat(i=q, j=out, index=index, mask=mask)
		elif self.label_head == 'sage':
			out = self.sage(j=out, index=index, mask=mask)

		return out

	def gcn(self, x_j, e, mask=None):
		if mask != None:
			inverted_mask = (torch.ones_like(mask) - mask).unsqueeze(-1)
			x_j = x_j * inverted_mask
		return x_j * e

	def gat(self, i, j, index, mask=None):
		x = torch.cat([i, j], dim=-1)
		alphas = self.lin_alpha(x)

		if mask != None:
			softmax_mask = mask * -1e6
			softmax_mask += torch.ones_like(softmax_mask)
			alphas = alphas * softmax_mask.unsqueeze(-1)

		alphas = softmax(alphas, index)

		out = j * alphas
		return out

	def sage(self, j, index, mask=None):
		node_degrees = degree(index).cpu()

		degree_spread = []
		for idx in range(node_degrees.shape[0]):
			d = int(node_degrees[idx].item())
			degree_spread += [d] * d

		degree_spread = torch.LongTensor(degree_spread).unsqueeze(-1).unsqueeze(-1).to(j.get_device())

		j /= degree_spread

		if mask != None:
			inverted_mask = (torch.ones_like(mask) - mask).unsqueeze(-1)
			j = j * inverted_mask

		return j
		

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}({self.in_channels}, '
				f'{self.out_channels}, heads={self.heads})')
