import torch
import config
from data import get_graph_data
from models.mlp import MLP
from models.gnn import GNN
from training import GraphTrainer
from logger import Logger

torch.manual_seed(0)


#l = Logger()
#l.load('logs/log.json')
#l.plot()
#l.plot_hyperparam_search('hyperparam_search.json')
#exit()



#data_dict = get_proteins_data_dicts()

#model = mlp(112, 64, 112, num_layers=5, dropout_p=0.2)

#criterion = torch.nn.BCEWithLogitsLoss()

#train(model, data_dict['train'], data_dict['valid'], criterion, num_epochs=200)

graph, split_idx = get_graph_data()

trainer = GraphTrainer(graph, split_idx, train_batch_size=1)
#trainer.normalise()
criterion = torch.nn.BCEWithLogitsLoss()


#model = MLP(trainer.graph.x.size(-1), 256, 112, num_layers=3, dropout=0.3)
model = GNN('TransformerConv', trainer.graph.x.size(-1), 64, 112, 1, 0.1)

trainer.train(model.to(config.device), criterion, num_epochs=300, lr=0.01, save_log=True, num_runs=10, use_scheduler=False)
#trainer.test(model, criterion, save_path='y_pred.pt')

#param_dict = {'lr':(1e-4,1e-1), 'layers':(1,7), 'hid_dim':(32,350), 'dropout':(0,0.5)}
#trainer.hyperparam_search(model=MLP, param_dict=param_dict, num_searches=50)

l = Logger()
l.load('logs/log.json')
l.plot()
