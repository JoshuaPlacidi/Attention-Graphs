import torch
from data import get_graph_data
from models.mlp import MLP
from models.gnn import GNN
from training import GraphTrainer


torch.manual_seed(0)


graph, split_idx = get_graph_data()

trainer = GraphTrainer(graph, split_idx, train_batch_size=32, train_neighbour_size=[100], valid_neighbour_size=[100], label_mask_p=0.8)
criterion = torch.nn.BCEWithLogitsLoss()

#mlp = MLP(trainer.graph.x.size(-1), 64, 112, num_layers=2, dropout=0.25)
gcn = GNN(conv_type='GCN', propagation='both', label_process='atn_embed', in_dim=16, hid_dim=64, out_dim=112, num_layers=2, dropout=0.25)
gat = GNN(conv_type='GAT', propagation='both', label_process='atn_embed', in_dim=16, hid_dim=64, out_dim=112, num_layers=2, dropout=0.25)
sage = GNN(conv_type='SAGE', propagation='both', label_process='atn_embed', in_dim=16, hid_dim=64, out_dim=112, num_layers=2, dropout=0.25)
tfc = GNN(conv_type='TFC', propagation='both', label_process='atn_embed', in_dim=16, hid_dim=64, out_dim=112, num_layers=2, dropout=0.25)


models = [gcn, gat, sage, tfc]

trainer.run_experiment(models, model_runs=5, num_epochs=200)
