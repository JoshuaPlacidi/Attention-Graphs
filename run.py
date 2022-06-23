import torch
from data import get_graph
from models.mlp import mlp
from models.gnn import GCN, SAGE
from training import train, GraphTrainer

#data_dict = get_proteins_data_dicts()

#model = mlp(112, 64, 112, num_layers=5, dropout_p=0.2)

#criterion = torch.nn.BCEWithLogitsLoss()

#train(model, data_dict['train'], data_dict['valid'], criterion, num_epochs=200)

graph_data, split_idx = get_graph()


trainer = GraphTrainer(graph_data, split_idx)
#trainer.normalise()
criterion = torch.nn.BCEWithLogitsLoss()

model = SAGE(graph_data.num_features, 350, 112, 4, 0.35)
#model = GCN(graph_data.num_features, 256, 112, 3, 0.15)

trainer.train(model, criterion, num_epochs=100, lr=2e-1, save_log=True)
trainer.test(model, criterion, save_path='y_pred.pt')

#param_dict = {'lr':(1e-4,1e-1), 'layers':(2,6), 'hid_dim':(128,512), 'dropout':(0,0.5)}
#trainer.hyperparam_search(model=SAGE, param_dict=param_dict, num_searches=50) 

