from tkinter import W
import torch
from tqdm import tqdm
from ogb.nodeproppred import Evaluator
from logger import Logger
import numpy as np 
import json
from torch_geometric.loader import DataLoader, NeighborLoader
import torch_geometric.transforms as T
from torch_scatter import scatter
import config

class GraphTrainer():
	'''
	Class for full batch graph training 
	'''
	def __init__(self, graph, split_idx, train_batch_size=64, evaluate_batch_size=64):
		'''
		params:
			- graph dataset
			- dictionary for storing the sample splits (train | valid | test) indexes
		'''
#		graph.num_nodes = torch.tensor(graph.num_nodes)
		self.graph = graph#.to(config.device)
		self.split_idx = split_idx
		self.evaluator = Evaluator(name='ogbn-proteins')

		# aggregate edge features using mean
		x = scatter(graph.edge_attr, graph.edge_index[0], dim=0, dim_size=graph.num_nodes, reduce='mean')
		self.graph.x = x
		# use node2vec embeddings
		# emb = torch.load('embedding.pt', map_location='cpu')
		# x = torch.cat([x, emb], dim=-1)
		
		#self.transforms = T.Compose([T.ToSparseTensor(remove_edge_index=False)])
		#self.graph = self.transforms(self.graph)

		self.train_batch_size = train_batch_size
		self.evaluate_batch_size = evaluate_batch_size
		
		# set feature variables
		self.train_loader = NeighborLoader(
								self.graph,
								num_neighbors=[10,5],
								batch_size=self.train_batch_size,
								directed=False,
								replace=True,
								shuffle=True,
								input_nodes=split_idx['train'],
								#transform=self.transforms,
		)
		
		self.valid_loader = NeighborLoader(
                                self.graph,
                                num_neighbors=[10,5],
                                batch_size=self.evaluate_batch_size,
								replace=True,
                                directed=False,
                                shuffle=False,
								input_nodes=split_idx['valid'],
                                #transform=self.transforms,
        )
		
	def normalise(self):
		'''
		normalise the graph for graph convolution calculation
		'''
		adj_t = self.graph.adj_t.set_diag()
		deg = adj_t.sum(dim=1).to(torch.float)
		deg_inv_sqrt = deg.pow(-0.5)
		deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
		adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
		self.graph.adj_t = adj_t
		
	def count_parameters(self, model):
		total_params = 0
		for _, parameter in model.named_parameters():
			if not parameter.requires_grad: 
				continue
			param = parameter.numel()
			total_params+=param
		return total_params

	def train(self, model, criterion, num_runs=1, num_epochs=10, lr=1e-3, use_scheduler=True, save_log=False, valid_step=5):
		'''
		train a model in full batch graph mode
		params:
			- model: PyTorch model to train
			- criterion: object to calculate loss between model predictions and targets
			- num_runs: number of runs of training to complete, model params are reset between runs
			- num_epochs: number of epochs to train for in each run
			- lr: initial learning rate
			- use_scheduler: whether to incremently decrease learning rate or not
			- save_log: if model logs should be saved to file
		returns:
			Logger object with logs of the total training cycle
		'''

		# store model and training information and save it in the logger
		info = model.param_dict
		info['num_runs'], info['batch_size'], info['lr'], info['num_epochs'], info['use_scheduler'], info['trainable_parameters'] = num_runs, self.train_batch_size, lr, num_epochs, use_scheduler, self.count_parameters(model)
		print('Training config: {0}'.format(info))
		logger = Logger(info=model.param_dict)

		# perform a new training experiement for each run, reseting the model parameters each time
		for run in range(1, num_runs+1):
			print('R' + str(run))

			# reset the model parameters
			model.reset_parameters()
			optimizer = torch.optim.Adam(model.parameters(), lr=lr)

			# define scheduler
			if use_scheduler:
				scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=2e-4, factor=0.1, cooldown=5, min_lr=1e-9)

			valid_loss, valid_roc = self.evaluate(model, sample_set='valid', criterion=criterion)

			epoch_bar = tqdm(range(1, num_epochs+1))
			for epoch in epoch_bar:
				# perform a train pass
				train_loss, train_roc = self.train_pass(model, optimizer, criterion)
				current_lr = optimizer.param_groups[0]['lr']

				results_dict = {}
				results_dict['run'], results_dict['epoch'], results_dict['lr'], results_dict['train_loss'], results_dict['train_roc'], results_dict['valid_loss'], results_dict['valid_roc'] = run, epoch, current_lr, train_loss, train_roc, valid_loss, valid_roc

				if epoch % valid_step == 0:
					# construct a results dictionary to store training parameters and model performance metrics
					valid_loss, valid_roc = self.evaluate(model, sample_set='valid', criterion=criterion)
					results_dict['valid_loss'], results_dict['valid_roc'] = valid_loss, valid_roc
				
				logger.log(results_dict)

				epoch_bar.set_description(
					"E {0}: LR({1}), T{2}, V{3}".format(
						epoch,
						round(current_lr,9),
						(round(results_dict['train_loss'],5), round(results_dict['train_roc'],5)),
						(round(results_dict['valid_loss'],5), round(results_dict['valid_roc'],5))
					))


				if use_scheduler:
					scheduler.step(results_dict['valid_loss'])

					# exit training if the learning rate drops to low
					if current_lr <= 1e-7:
						break

		logger.print()

		# save logs files
		if save_log:
			logger.save("logs/log.json".format(run))

		return logger


	def train_pass(self, model, optimizer, criterion):
		'''
		pass full graph through model and update weights
		params:
			- model: PyTorch model to train
			- optimizer: optimizer to use to update weights
			- criterion: object to calculate loss between target and model output
		returns:
			Float of loss of the model on the train set
		'''
		model.train()
		total_loss, count = 0, 0
		pred = []
		gts = []

		for batch in self.train_loader:
			optimizer.zero_grad()

			# calculate output
			pred_y = model(batch.to(config.device))[:batch.batch_size]

			pred.append(pred_y.cpu())
			gts.append(batch.y[:batch.batch_size])

			# update weights
			loss = criterion(pred_y, batch.y[:batch.batch_size].to(torch.float))
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			count += 1

		train_loss = total_loss / count
		train_roc = self.evaluator.eval({
									'y_true': torch.cat(gts, dim=0),
									'y_pred': torch.cat(pred, dim=0),
								})['rocauc']

		return train_loss, train_roc
			
		

	def evaluate(self, model, sample_set='valid', criterion=torch.nn.BCEWithLogitsLoss(), save_path=None):
		'''
		perform a evaluation of a model on validation set
		params:
			- model: model to evaluate
			- criterion: object to calculate loss between target and model output
			- save_path (optional): if provided the complete y_pred output will be stored at this file location
		returns:
			Dictionary object containing the results from test pass
		'''
		with torch.no_grad():
			model.eval()

			if sample_set == 'valid':
				sample_loader = self.valid_loader
			else:
				raise Exception('trainer.evaluate(): sample_set "' + sample_set + '" not recognited')
				
			pred, count = [], 0
			for batch in sample_loader:
				pred_y = model(batch.to(config.device))[:batch.batch_size]
				loss = criterion(pred_y, batch.y[:batch.batch_size].to(torch.float)).item()
				pred.append(pred_y.cpu())
				count += 1

			pred = torch.cat(pred, dim=0)

			# loop over each sample set (train | valid | test) and calculate loss and ROC
			loss = loss / count
			roc = self.evaluator.eval({
									'y_true': self.graph.y[self.split_idx[sample_set]],
									'y_pred': pred,
								})['rocauc']
		
			if save_path:
				torch.save(pred, save_path)	

		return loss, roc

	def hyperparam_search(self, model, param_dict, criterion=torch.nn.BCEWithLogitsLoss(), num_searches=10):
		'''
		performs a hyperparameter search over a range of values, each search randomly selects
		values from each parameters specified range
		params:
			- model: uninitialised model object to run search on
			- param_dict: a dictionary with keys of parameters and values of their search ranges
			- criterion: method to evaluate model loss
			- num_searches: how many hyperparamet searches to run
		'''
		param_types = ['lr', 'hid_dim', 'layers', 'dropout']
		assert set(param_dict.keys()) == set(param_types)

		# define variables for storing log information and keeping track of the best parameters
		logs = []
		best_loss = 1000
		best_params = {}

		# for each search: initialise a model with hyperparameters and evaluate its performance
		for search in range(num_searches):

			# dictionary to store the sampled hyperparams for this search
			params = {}

			# for each param uniformly sample from its range
			for p in param_types:

				value = np.random.uniform(param_dict[p][0], param_dict[p][1], 1)[0]
				
				# convert the value to an int if nessassary
				if p == 'hid_dim' or p == 'layers': value = int(value)

				params[p] = value

			print('S {0}/{1}'.format(search, num_searches))

			# initialise a modle with sample hyperparameters
			m = model(in_dim=self.graph.num_features, hid_dim=params['hid_dim'], out_dim=112,
						num_layers=params['layers'], dropout=params['dropout'])

			# initialise training strategy with sampled hyperparameters
			m_logger = self.train(m, criterion, num_epochs=200, lr=params['lr'], save_log=True)
			logs.append(m_logger.logs)
			
			# if model is best so far, save its parameters
			m_loss = min(m_logger.logs['valid_loss'])
			if m_loss < best_loss:
				best_loss = m_loss
				best_params = params
			
			# store hyperparamet logs
			with open('hyperparam_search.json', 'w') as fp:
				json.dump(logs, fp)

		# print results
		print('Best Params:', best_params, ' with best loss:', best_loss)






