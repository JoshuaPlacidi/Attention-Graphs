import torch
from tqdm import tqdm
from ogb.nodeproppred import Evaluator

evaluator = Evaluator(name = 'ogbn-proteins')

def train(model, train_dataloader, val_dataloader, criterion, num_epochs=10):

	optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
	results_dict = {}	

	eval_results = evaluate(model, val_dataloader, criterion)
	eval_results['train_loss'] = -1
	
	results_dict['0'] = eval_results

	epoch_pbar = tqdm(range(num_epochs))

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

		for batch in tqdm(dataloader):
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
