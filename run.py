import torch
from data import get_graph
from models.mlp import MLP
from models.gnn import GCN, SAGE
from training import train, GraphTrainer
from logger import Logger

#l = Logger()
#l.load('logs/log.json')
#l.plot()
#l.plot_hyperparam_search('hyperparam_search.json')
#exit()




#data_dict = get_proteins_data_dicts()

#model = mlp(112, 64, 112, num_layers=5, dropout_p=0.2)

#criterion = torch.nn.BCEWithLogitsLoss()

#train(model, data_dict['train'], data_dict['valid'], criterion, num_epochs=200)

graph_data, split_idx = get_graph()


trainer = GraphTrainer(graph_data, split_idx)
#trainer.normalise()
criterion = torch.nn.BCEWithLogitsLoss()

#model = SAGE(graph_data.num_features, 309, 112, 1, 0.1)
#model = GCN(graph_data.num_features, 256, 112, 3, 0.15)
model = MLP(trainer.graph.x.size(-1), 256, 112, num_layers=3, dropout=0.5)

trainer.train(model, criterion, num_epochs=300, lr=0.01, save_log=True, num_runs=10, use_scheduler=False)
#trainer.test(model, criterion, save_path='y_pred.pt')

#param_dict = {'lr':(1e-4,1e-1), 'layers':(1,7), 'hid_dim':(32,350), 'dropout':(0,0.5)}
#trainer.hyperparam_search(model=MLP, param_dict=param_dict, num_searches=50)

l = Logger()
l.load('logs/log.json')
l.plot()
