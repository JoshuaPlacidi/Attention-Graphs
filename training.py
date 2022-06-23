import torch
from tqdm import tqdm
from ogb.nodeproppred import Evaluator
from logger import Logger
import numpy as np 
import json
import random

def train(model, train_dataloader, val_dataloader, criterion, num_epochs=10):

	optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
	results_dict = {}	

	print("Calculating initial performance on validation set")
	eval_results = evaluate(model, val_dataloader, criterion)
	eval_results['train_loss'] = -1
	
	results_dict['0'] = eval_results

	epoch_pbar = tqdm(range(num_epochs))

	print("Running training for %s epochs", num_epochs)
	for epoch in epoch_pbar:

		model.train()
		for batch in train_dataloader:
			y = batch[-1]
			pred = model(*batch[:-1])

			loss = criterion(pred, y.float())
			loss.backward()
			optimizer.step()
			model.zero_grad()

		epoch_results = {}
		train_results = evaluate(model, train_dataloader, criterion)
		epoch_results['train_rocauc'] = train_results['rocauc'] 
		epoch_results['train_loss'] = train_results['loss']
		
		
		val_results = evaluate(model, val_dataloader, criterion)
		epoch_results['val_rocauc'] = val_results['rocauc']
		epoch_results['val_loss'] = val_results['loss']
	
		epoch_pbar.set_description(
			"Epoch %s: train loss %s, train rocauc %s, val loss %s, val rocauc %s" % 
			tuple([round(x,3) for x in [epoch, epoch_results['train_loss'], epoch_results['train_rocauc'], epoch_results['val_loss'], epoch_results['val_rocauc']]])
			)
		results_dict[str(epoch)] = epoch_results

	print(results_dict)
	process_results(results_dict)

def process_results(results_dict):
	best_val_loss = results_dict[list(results_dict.keys())[0]]['val_loss']
	best_epoch = '0'

	for epoch_str, result in results_dict.items():
		if result['val_loss'] < best_val_loss:
			best_val_loss = result['val_loss']
			best_epoch = epoch_str

	print("Best model (epoch %s):\n" % (best_epoch), results_dict[best_epoch])

def evaluate(model, dataloader, criterion):
	with torch.no_grad():

		model.eval()
		running_loss = 0
		pred_all = torch.Tensor([])
		y_all = torch.Tensor([])

		for batch in dataloader:
			y = batch[-1]
			y_all = torch.cat([y_all, y])

			pred = model(*batch[:-1])
			pred_all = torch.cat([pred_all, pred])			

			loss = criterion(pred, y.float())
			running_loss += loss

		eval_dict = {"y_true":y_all, "y_pred":pred_all}
		eval_results = evaluator.eval(eval_dict)	

		eval_results['loss'] = running_loss.item() / len(dataloader)

		return eval_results	


class GraphTrainer():
	def __init__(self, graph, split_idx):
		self.graph = graph
		self.split_idx = split_idx
		self.evaluator = Evaluator(name='ogbn-proteins')

	def normalise(self):
		adj_t = self.graph.adj_t.set_diag()
		deg = adj_t.sum(dim=1).to(torch.float)
		deg_inv_sqrt = deg.pow(-0.5)
		deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
		adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
		self.graph.adj_t = adj_t
		
	def train(self, model, criterion, runs=1, num_epochs=10, lr=1e-3, use_scheduler=True, save_log=False):
		torch.manual_seed(0)

		logger = Logger(info=model.param_dict)
		for run in range(1, runs+1):
			print('R' + str(run))
			model.reset_parameters()
			optimizer = torch.optim.Adam(model.parameters(), lr=lr)

			if use_scheduler:
				scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-9, cooldown=7)


			epoch_bar = tqdm(range(1, num_epochs+1))
			for epoch in epoch_bar:
				loss = self.train_pass(model, optimizer, criterion)
				
				results_dict = self.test(model, criterion)
				results_dict['run'] = run
				results_dict['epoch'] = epoch
				current_lr = optimizer.param_groups[0]['lr']
				results_dict['lr'] = current_lr
				logger.log(results_dict)

				epoch_bar.set_description(
					"E {0}, LR {1}, T{2}, V{3}".format(epoch, round(current_lr,6),
						*[(results_dict[s]['loss'], results_dict[s]['roc']) for s in ['train','valid']])
					)

				scheduler.step(results_dict['valid']['loss'])

		if save_log:
			logger.save("log.json")
			logger.plot("plot.eps")
		
		return logger.logs 

	def train_pass(self, model, optimizer, criterion):
		model.train()
		optimizer.zero_grad()

		out = model(self.graph.x, self.graph.adj_t)[self.split_idx['train']]

		loss = criterion(out, self.graph.y[self.split_idx['train']].to(torch.float))
		loss.backward()
		optimizer.step()

		return loss.item()
			
		
	def test(self, model, criterion, save_path=None):
		with torch.no_grad():
			model.eval()
				
			y_pred = model(self.graph.x, self.graph.adj_t)

			results_dict = {}
			for s in ['train', 'valid', 'test']:
				loss = criterion(
							y_pred[self.split_idx[s]],
							self.graph.y[self.split_idx[s]].to(torch.float)
						).item()
				roc = self.evaluator.eval({
									'y_true': self.graph.y[self.split_idx[s]],
									'y_pred': y_pred[self.split_idx[s]],
								})['rocauc']

				results_dict[s] = {'loss':round(loss,3), 'roc':round(roc,3)}
		
			if save_path:
				torch.save(y_pred, save_path)	

		return results_dict

	def hyperparam_search(self, model, param_dict, criterion=torch.nn.BCEWithLogitsLoss(), num_searches=10):
		param_types = ['lr', 'hid_dim', 'layers', 'dropout']
		assert set(param_dict.keys()) == set(param_types)

		logs = []
		
		best_loss = 1000
		best_params = {}

		for search in range(num_searches):
			params = {}
			for p in param_types:

				if p == 'lr':
					value = random.choice([1e-1, 1e-2, 1e-3, 1e-4])
					value = value * np.random.uniform(1, 9, 1)[0]
				else:
					value = np.random.uniform(param_dict[p][0], param_dict[p][1], 1)[0]

				if p == 'hid_dim' or p == 'layers': value = int(value)
				params[p] = value

			print('S {0}/{1}: {2}'.format(search, num_searches, params))
			m = model(in_dim=self.graph.num_features, hid_dim=params['hid_dim'], out_dim=112,
						num_layers=params['layers'], dropout=params['dropout'])

			m_log = self.train(m, criterion, num_epochs=100, lr=params['lr'], save_log=True)
			logs.append(m_log)
			
			m_loss = min(m_log['valid_loss'])
			if m_loss < best_loss:
				best_loss = m_loss
				best_params = params
			
			with open('hyperparam_search.json', 'w') as fp:
				json.dump(logs, fp)


		print('Best Params:', params, ' with best loss:', best_loss)




