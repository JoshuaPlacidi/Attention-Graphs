import torch
import json
import matplotlib.pyplot as plt
from collections import defaultdict

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
			

	def print(self, run=None):
		if run is not None:
			return
		return

	def print_statistics(self, run=None):
		if run is not None:
			result = 100 * torch.tensor(self.results[run])
			argmax = result[:, 1].argmax().item()
			print(f'Run {run + 1:02d}:')
			print(f'Highest Train: {result[:, 0].max():.2f}')
			print(f'Highest Valid: {result[:, 1].max():.2f}')
			print(f'  Final Train: {result[argmax, 0]:.2f}')
			print(f'   Final Test: {result[argmax, 2]:.2f}')
		else:
			result = 100 * torch.tensor(self.results)

			best_results = []
			for r in result:
				train1 = r[:, 0].max().item()
				valid = r[:, 1].max().item()
				train2 = r[r[:, 1].argmax(), 0].item()
				test = r[r[:, 1].argmax(), 2].item()
				best_results.append((train1, valid, train2, test))

			best_result = torch.tensor(best_results)

			print(f'All runs:')
			r = best_result[:, 0]
			print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
			r = best_result[:, 1]
			print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
			r = best_result[:, 2]
			print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
			r = best_result[:, 3]
			print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
