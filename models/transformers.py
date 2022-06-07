import torch
import torch.nn as nn

# tranformer models

class transformer(nn.Module):
	def __init__(self, input_size, hidden_size, out_size,
				 data_src=None, num_heads=8, num_layers=6, dropout_p=0.0):
		super(transformer, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.out_size = out_size

		self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		self.sos_embedding = nn.Embedding(2, hidden_size)

		self.e_to_hid = nn.Linear(8, hidden_size)
		self.l_to_hid = nn.Linear(112, hidden_size)

	def forward(self, e, l):
		sos = self.sos_embeddings(0)
		
		e = self.e_to_hid(e)
		l = self.l_to_hid(l)

		x = e + l
		x = torch.cat((sos, x), 1)

		x = self.encoder(x)

		return out
