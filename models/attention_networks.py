import torch
import torch.nn as nn

class transformer(nn.Module):
	def __init__(self, input_size, hidden_size, out_size,
				 num_heads=8, num_layers=6, dropout_p=0.0):
		super(transformer, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.out_size = out_size

		self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		self.sos_embedding = nn.Embedding(2, hidden_size)

	def forward(self, x):
		sos = torch.zeros(self.hidden_size)
		
		return out
