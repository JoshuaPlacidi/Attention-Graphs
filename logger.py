import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools


class Logger(object):
	def __init__(self, info=None):
		self.logs = defaultdict(list)
		self.logs['info'] = info

		self.train_col = 'lightseagreen'
		self.valid_col = 'mediumslateblue'
		self.test_col = 'orangered'
		self.alt_col = 'crimson'

	def log(self, results_dict):
		#TODO Adapt logs to properly store values over multiple runs
		sample_sets = ['train', 'valid', 'test']

		for k, v in results_dict.items():
			if k in sample_sets:
				self.logs[k + '_loss'].append(results_dict[k]['loss'])
				self.logs[k + '_roc'].append(results_dict[k]['roc'])
			else:
				self.logs[k].append(v)
		

	def save(self, filepath):
		assert filepath.endswith('.json')
		with open(filepath, 'w') as fp:
			json.dump(self.logs, fp)	

	def plot(self, filepath, show_test=False):
		assert filepath.endswith('.eps')

		fig = plt.figure(figsize=(14, 6.5), dpi=80)
		
		fig.add_subplot(121)
		tl = plt.plot(self.logs['train_loss'], c=self.train_col, label='train')
		vl = plt.plot(self.logs['valid_loss'], c=self.valid_col, label='valid')
		if show_test:
			plt.plot(self.logs['test_loss'], c=self.test_col, label='test')
		plt.ylabel('Loss')
		plt.ylim(0, 1)
		plt.title('Loss Curves and Learning Rate')

		lr = plt.gca().twinx().plot(self.logs['lr'], c=self.alt_col, label='LR')
		plt.yscale('log')
		plt.ylabel('Learning Rate')
		
		lns = tl + vl + lr
		labs = [l.get_label() for l in lns]
		plt.legend(lns, labs, loc=0)


		fig.add_subplot(122)
		plt.plot(self.logs['train_roc'], c=self.train_col, label='train')
		plt.plot(self.logs['valid_roc'], c=self.valid_col, label='valid')
		if show_test:
			plt.plot(self.logs['test_roc'], c=self.test_col, label='test')
		plt.xlabel('Epoch')
		plt.ylabel('Reciever Operator Curve')
		plt.title('ROC')
		plt.legend()
		#fig.add_subplot(133)

		#plt.plot(self.logs['train_loss'], c=self.train_col)
		#plt.plot(self.logs['valid_loss'], c=self.valid_col)	

		#plt.gca().twinx().plot(self.logs['lr'], c=self.alt_col)
		#plt.yscale('log')
		#plt.ylabel('Learning Rate')		
		#plt.title('LR')
		fig.tight_layout()
		plt.savefig(filepath, format='eps')
			
	def plot_hyperparam_search(self, filepath):
		with open(filepath) as json_file:
			hyperparam_logs = json.load(json_file)
		
		params = defaultdict(list)
		score = []

		for log in hyperparam_logs:

			for k, v in log['info'].items():
				params[k].append(v)
#			params['lr'].append(log['lr'][0])
			score.append(max(log['valid_roc']))

		for p in params.keys():
#			print(p)
#			print(params[p])
			plt.scatter(params[p], score)
			plt.title(p)
			plt.xlabel(p)
			plt.ylabel('valid roc')
			plt.ylim(0,1)
#			plt.savefig(p + '.eps', format='eps')	
			plt.show()

	def print(self, run=None):
		runs = self.logs['run']
		num_runs = max(runs)

		train_losses, train_rocs = [], []
		valid_losses, valid_rocs = [], []

		for r in range(1, num_runs+1):
			run_start = runs.index(r)
			run_end = len(runs) - runs[::-1].index(r) - 1

			best_idx = min(
				range(len(self.logs['valid_loss'][run_start:run_end+1])),
				key=self.logs['valid_loss'][run_start:run_end+1].__getitem__
			)

			train_losses.append(self.logs['train_loss'][run_start:run_end+1][best_idx])
			train_rocs.append(self.logs['train_roc'][run_start:run_end+1][best_idx])
			valid_losses.append(self.logs['valid_loss'][run_start:run_end+1][best_idx])
			valid_rocs.append(self.logs['valid_roc'][run_start:run_end+1][best_idx])

		print('Results from {0} runs'.format(num_runs))
		print('Train mean loss {0} +/- {1}'.format(np.mean(train_losses), np.std(train_losses)))
		print('Train mean roc  {0} +/- {1}'.format(np.mean(train_rocs), np.std(train_rocs)))
		print('Valid mean loss {0} +/- {1}'.format(np.mean(valid_losses), np.std(valid_losses)))
		print('Valid mean roc  {0} +/- {1}'.format(np.mean(valid_rocs), np.std(valid_rocs)))

	def load(self, filepath):
		with open(filepath) as fp:
			self.logs = json.load(fp)
	
	def plot(self):
		runs = self.logs['run']
		num_runs = max(runs)
		
		if num_runs > 1: # if more than 1 run in logs then plot means

			fig = plt.figure(figsize=(14, 6.5), dpi=80)

			losses = []

			for r in range(1, num_runs+1):
				run_start = runs.index(r)
				run_end = len(runs) - runs[::-1].index(r) - 1
				run_valid_loss = self.logs['valid_loss'][run_start:run_end+1]				

				losses.append(run_valid_loss)
				plt.plot(
					range(0, run_end - run_start + 1), 
					run_valid_loss,
					color="lightgrey"
				)

			epoch_loss = list(map(list, itertools.zip_longest(*losses, fillvalue=None)))
			epoch_loss[2].append(None)
			
			for e in range(len(epoch_loss)):
				epoch_loss[e] = [l for l in epoch_loss[e] if l is not None]

			mean_epoch_loss = list(map(np.mean, epoch_loss))

			plt.plot(range(0, max(self.logs['epoch'])), mean_epoch_loss)
			plt.show()

		else:
			raise('plot need more runs')










