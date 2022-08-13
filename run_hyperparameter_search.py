import torch
import config
from data import get_graph_data
from models.mlp import MLP
from models.gnn import GNN
from models.multiprop import MultiPropGNN
from training import GraphTrainer
from logger import Logger

torch.manual_seed(0)

graph, split_idx = get_graph_data()

trainer = GraphTrainer(graph, split_idx, train_batch_size=32, evaluate_batch_size=32, train_neighbour_size=[100,1,1], valid_neighbour_size=[100,1,1], label_mask_p=0.8)#0.126)
criterion = torch.nn.BCEWithLogitsLoss()

model = MultiPropGNN
param_dict = {'lr':(0,5), 'layers':(1,6), 'hid_dim':(32,350), 'dropout':(0,0.7)}
#model = model(in_dim=8, hid_dim=237, out_dim=112, num_layers=2, dropout=0.1258)
#trainer.train(model, criterion = criterion, num_epochs=2, lr = 0.0001, save_log=True)
trainer.hyperparam_search(model=model, param_dict=param_dict, num_searches=3, num_epochs=3)