import torch
import config
from data import get_graph_data
from models.mlp import MLP
from models.gnn import GNN
from models.transformers import AttentionGNN
from training import GraphTrainer
from logger import Logger

torch.manual_seed(0)


graph, split_idx = get_graph_data()

trainer = GraphTrainer(graph, split_idx, train_batch_size=32, sampler_num_neighbours=100, label_mask_p=0.0)
criterion = torch.nn.BCEWithLogitsLoss()

mlp = MLP(trainer.graph.x.size(-1), 64, 112, num_layers=3, dropout=0.25)
gcn = GNN(conv_type='GCN', propagation='feature', in_dim=8, hid_dim=64, out_dim=112, num_layers=2, dropout=0.25)
gat = GNN(conv_type='GAT', propagation='feature', in_dim=8, hid_dim=64, out_dim=112, num_layers=2, dropout=0.25)
sage = GNN(conv_type='SAGE', propagation='feature', in_dim=8, hid_dim=64, out_dim=112, num_layers=2, dropout=0.25)
tfc = GNN(conv_type='TFC', propagation='feature', in_dim=8, hid_dim=64, out_dim=112, num_layers=2, dropout=0.25)


models = [mlp, gcn, sage, gat, tfc]

trainer.run_experiment(models, model_runs=10, num_epochs=200)
