import torch
from data import get_proteins_data_dict
from models.mlp import mlp
from training import train

data_dict = get_proteins_data_dict()

model = mlp(8, 64, 112, num_layers=5, dropout_p=0.2)

criterion = torch.nn.BCEWithLogitsLoss()

train(model, data_dict['train'], data_dict['val'], criterion, num_epochs=200)

