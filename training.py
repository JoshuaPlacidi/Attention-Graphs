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
	Class for subgraph batch training
	'''
	def __init__(self,
			  graph,
			  split_idx,
			  train_batch_size=64,
			  evaluate_batch_size=None,
			  label_mask_p=0.5,
			  train_neighbour_size=[100],
			  valid_neighbour_size=[100],
		):
		'''
		params:
			- graph: pytorch geometric graph object
			- split_idx: dictionary for storing the sample splits (train | valid | test) indexes
			- train_batch_size: the batch size to be used for training
			- evaluate_batch_size: the batch size to be used when evaluating a model
			- label_mask_p: the probability of masking out node labels at training time
			- train_neighbour_size: list of number of nodes to sample at each neighbour order when training
			- valid_neighbour_Size: list of number of nodes to sample at each neighbour order during validation
		'''

		self.graph = graph
		self.split_idx = split_idx
		self.evaluator = Evaluator(name='ogbn-proteins')
		self.train_neighbour_size = train_neighbour_size
		self.valid_neighbour_size = valid_neighbour_size
		self.label_mask_p = label_mask_p

		# aggregate edge features using mean
		features = scatter(graph.edge_attr, graph.edge_index[0], dim=0, dim_size=graph.num_nodes, reduce='mean')
		self.graph.features = features
		
		# mask labels
		self.graph.train_mask = self.label_mask(label_mask_p, mask_eval=True)
		self.graph.eval_mask = self.label_mask(0, mask_eval=True)

		# set batch sizes
		self.train_batch_size = train_batch_size
		self.evaluate_batch_size = evaluate_batch_size if evaluate_batch_size else train_batch_size
		
		# create sub graph sampler objects
		self.train_loader = NeighborLoader(
								self.graph,
								num_neighbors=self.train_neighbour_size,
								batch_size=self.train_batch_size,
								directed=True,
								replace=True,
								shuffle=True,
								input_nodes=split_idx['train'],
		)
		
		self.valid_loader = NeighborLoader(
								self.graph,
								num_neighbors=self.valid_neighbour_size,
								batch_size=self.evaluate_batch_size,
								replace=True,
								directed=True,
								shuffle=False,
								input_nodes=split_idx['valid'],
		)

		self.test_loader = NeighborLoader(
								self.graph,
								num_neighbors=[-1, 1],
								batch_size=1,
								replace=True,
								directed=True,
								shuffle=False,
								input_nodes=split_idx['test'],
		)
	
	def label_mask(self, label_mask_p, mask_eval=True):
		'''
		Create a label mask for each node, where 0 means no mask is applied and 1 masks out that nodes label
		params:
			- label_mask_p: the probability of masking out a nodes label
			- mask_eval: if True all validation and testing nodes have their labels masked, if False they are masked with label_mask_p probability
		
		returns:
			- mask: tensor where a 0 at index i means node i will not be masked, a 1 at index i means node i will be masked
		'''
		if not mask_eval:
			raise NotImplemented('unmasked valid and test labels is not implmented')

		# randomly select training points to keep (0) and remove(1)
		train_labels = self.graph.y[self.split_idx['train']]
		inverted_label_mask_p = 1-label_mask_p
		train_mask = torch.rand(train_labels.shape[0]).ge(inverted_label_mask_p).unsqueeze(-1)
		
		# remove ALL valid and test labels
		valid_test_mask = torch.ones(len(self.split_idx['valid']) + len(self.split_idx['test']), 1)

		# create mask and inverted mask
		mask = torch.cat((train_mask, valid_test_mask), 0)

		return mask
	
	def count_parameters(self, model):
		'''
		Counts the number of trainable parameters in a pytorch model
		params:
			model: torch.nn.Module object to count the number of parameters of

		returns:
			total_params: integer of number of parameters
		'''
		total_params = 0
		for _, parameter in model.named_parameters():

			# only count parameters that are trainable
			if not parameter.requires_grad:
				continue

			param = parameter.numel()
			total_params+=param
		return total_params

	def train(
			self,
			model,
			criterion,
			num_runs=1,
			num_epochs=10,
			lr=1e-3,
			use_scheduler=True,
			save_log=False,
			valid_step=5,
			return_state_dict=False
		):
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
		torch.manual_seed(0)
		model.to(config.device)

		# store model and training information and save it in the logger
		info = model.param_dict
		info['num_runs'] = num_runs
		info['batch_size'] = self.train_batch_size
		info['train_neighbour_size'] = self.train_neighbour_size
		info['valid_neighbour_size'] = self.valid_neighbour_size
		info['label_mask_probability'] = self.label_mask_p
		info['trainable_parameters'] = self.count_parameters(model)
		info['lr'] = lr
		info['num_epochs'] = num_epochs
		info['use_scheduler'] = use_scheduler

		print('Training config: {0}'.format(info))
		logger = Logger(info=model.param_dict)

		if return_state_dict:
			best_state_dict = None
			best_loss_across_runs = 10000
		

		# perform a new training experiement for each run, reseting the model parameters each time
		for run in range(1, num_runs+1):
			print('R' + str(run))

			# reset the model parameters
			model.reset_parameters()
			optimizer = torch.optim.Adam(model.parameters(), lr=lr)

			# define scheduler
			if use_scheduler:
				scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=2e-4, factor=0.1, cooldown=5, min_lr=1e-9)

			valid_loss, valid_roc = None, None

			epoch_bar = tqdm(range(1, num_epochs+1))
			for epoch in epoch_bar:

				# generate new label masks
				self.graph.train_mask = self.label_mask(self.label_mask_p, mask_eval=True)
				self.graph.eval_mask = self.label_mask(0, mask_eval=True)

				# perform a train pass
				train_loss, train_roc = self.train_pass(model, optimizer, criterion)
				current_lr = optimizer.param_groups[0]['lr']

				# save results to dictionary object
				results_dict = {}
				results_dict['run'] = run
				results_dict['epoch'] = epoch
				results_dict['lr'] = current_lr
				results_dict['train_loss'] = train_loss
				results_dict['train_roc'] = train_roc
				results_dict['valid_loss'] = valid_loss
				results_dict['valid_roc'] = valid_roc

				if epoch % valid_step == 0 or epoch == 1:
					# construct a results dictionary to store training parameters and model performance metrics
					valid_loss, valid_roc = self.evaluate(model, sample_set='valid', criterion=criterion)
					results_dict['valid_loss'], results_dict['valid_roc'] = valid_loss, valid_roc

					if return_state_dict and valid_loss < best_loss_across_runs:
						best_state_dict = model.state_dict()
				
				# log results
				logger.log(results_dict)

				# update tqdm bar
				epoch_bar.set_description(
					"E {0}: LR({1}), T{2}, V{3}".format(
						epoch,
						round(current_lr,9),
						(round(results_dict['train_loss'],5), round(results_dict['train_roc'],5)),
						(round(results_dict['valid_loss'],5), round(results_dict['valid_roc'],5))
					))

				# if using shceduler then update it
				if use_scheduler:
					scheduler.step(results_dict['valid_loss'])

					# exit training if the learning rate drops to low
					if current_lr <= 1e-6:
						break


				# save logs files
				if save_log:
					logger.save("logs/{0}_log.json".format(info['model_type']))

		# print logs
		logger.print()

		if return_state_dict:
			return logger, best_state_dict
		else:
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

		# initialise evaluation evaluation trackers
		total_loss, count = 0, 0
		pred = []
		gts = []

		for batch in self.train_loader:
			optimizer.zero_grad()

			ground_truths = batch.y[:batch.batch_size].clone()

			# remove ground truth labels from batch to prevent label leakage
			batch.y[:batch.batch_size] = torch.zeros_like(batch.y[:batch.batch_size])

			# calculate output
			pred_y = model(batch)[:batch.batch_size].cpu()

			# store predictions and ground truths
			pred.append(pred_y)
			gts.append(ground_truths)

			# update weights
			loss = criterion(pred_y, ground_truths.float()).to(torch.float)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			count += 1

		# evaluate the model predictions against ground truths

		train_loss = total_loss / count
		train_roc = self.evaluator.eval({
									'y_true': torch.cat(gts, dim=0),
									'y_pred': torch.cat(pred, dim=0),
								})['rocauc']

		return train_loss, train_roc
			
		

	def evaluate(
		self,
		model,
		sample_set='valid',
		criterion=torch.nn.BCEWithLogitsLoss(),
		save_path=None
		):
		'''
		perform a evaluation of a model on validation set
		params:
			- model: model to evaluate
			- sample_set: the set of data to evaluate model perfomance on, 'train', 'test', 'valid'
			- criterion: object to calculate loss between target and model output
			- save_path (optional): if provided the complete y_pred output will be stored at this file location

		returns:
			Dictionary object containing the results from test pass
		'''
		with torch.no_grad():
			model.eval()

			if sample_set == 'valid':
				sample_loader = self.valid_loader
			elif sample_set == 'test':
				sample_loader = tqdm(self.test_loader)
			else:
				raise Exception('trainer.evaluate(): sample_set "' + sample_set + '" not recognited/implemented')
			
			# initialise evaluation trackers
			pred = []
			gts = []
			loss = 0
			count = 0

			# loop through sample loader batches
			for batch in sample_loader:

				ground_truths = batch.y[:batch.batch_size].clone()

				# remove ground truth labels from batch to prevent label leakage
				batch.y[:batch.batch_size] = torch.zeros_like(batch.y[:batch.batch_size])

				# calculate model prediction and loss
				pred_y = model(batch.to(config.device))[:batch.batch_size].cpu()
				loss += criterion(pred_y, ground_truths.float()).item()
				
				pred.append(pred_y)
				gts.append(ground_truths)

				count += 1

			# loop over each sample set (train | valid | test) and calculate loss and ROC
			loss = loss / count
			roc = self.evaluator.eval({
									'y_true': torch.cat(gts, dim=0),
									'y_pred': torch.cat(pred, dim=0),
								})['rocauc']
		
			# save model predictins
			if save_path:
				torch.save(pred, save_path)	

		return loss, roc

	def hyperparam_search(
			self,
			model,
			param_dict,
			criterion=torch.nn.BCEWithLogitsLoss(),
			num_searches=10,
			num_epochs=200,
			):
		'''
		performs a hyperparameter search over a range of values, each search randomly selects
		values from each parameters specified range
		params:
			- model: uninitialised model object to run search on
			- param_dict: a dictionary with keys of parameters and values of their search ranges
			- criterion: method to evaluate model loss
			- num_searches: how many hyperparamet searches to run
			- num_epochs: number of epochs to use for each search

		returns:
			None
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
				if p == 'hid_dim' or p == 'layers':
					value = int(value)
				elif p == 'lr':
					value = float('0.' + ('0' * int(value)) + str(int(np.random.uniform(0, 9, 1)[0])))

				params[p] = value

			print('S {0}/{1}'.format(search, num_searches))

			print('Test parameters:', params)
			# initialise a modle with sample hyperparameters
			m = model(in_dim=8, hid_dim=params['hid_dim'], out_dim=112, num_layers=params['layers'], dropout=params['dropout'])

			# initialise training strategy with sampled hyperparameters
			m_logger, state_dict = self.train(m, criterion, num_epochs=num_epochs, lr=params['lr'], save_log=True, return_state_dict=True)
			logs.append(m_logger.logs)
			
			# if model is best so far, save its parameters
			m_loss = min(m_logger.logs['valid_loss'])
			if m_loss < best_loss:
				best_loss = m_loss
				best_params = params
				torch.save(state_dict, 'best_model.pt')
			
			# store hyperparameter logs
			with open('hyperparam_search.json', 'w') as fp:
				json.dump(logs, fp)

		# print results
		print('Best Params:', best_params, ' with best loss:', best_loss)

		best_model = model(in_dim=8, hid_dim=best_params['hid_dim'], out_dim=112, num_layers=best_params['layers'], dropout=best_params['dropout'])
		best_model.load_state_dict(torch.load('best_model.pt'))
		best_model.eval()
		best_model.to(config.device)
		test_loss, test_roc = self.evaluate(best_model, sample_set='test', criterion=criterion)

		print('test_loss', test_loss)
		print('test_roc', test_roc)

	
	def run_experiment(
			self,
			models,
			model_runs=1,
			num_epochs=200,
			lr=0.001,
			criterion=torch.nn.BCEWithLogitsLoss(),
			):
		'''
		takes a set of models and runs them for model_runs amount of times and logs performance
		params:
			- models: list of model objects to run experiments on
			- model_runs: number of times to run each model
			- num_epochs: number of epochs for each run
			- lr: initial learning rate
			- criterion: criterion used for calculating loss

		returns:
			None
		'''
		
		logs = []

		# for each model in list
		for i, m in enumerate(models):
			print('E{0}'.format(i))
			
			# run model
			m_logger = self.train(m, criterion, num_epochs=num_epochs, lr=lr, save_log=False, num_runs=model_runs)

			# save logs to file
			logs.append(m_logger.logs)
			with open('logs/experiment_logs.json', 'w') as fp:
				json.dump(logs, fp)






