import os
import csv
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


now = datetime.now()


def train(
		train_loader: DataLoader,
		valid_loader: DataLoader,
		model: nn.Module,
		optimizer: torch.optim.Optimizer,
		loss_fn,
		metrics: list,
		num_epoch: int,
		save_dir: str  # save_domain/model_name/
):
	# save_domain/model_name/datetime/
	save_dir = os.path.join(save_dir, now.strftime('%Y-%m-%d-%H-%M'))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		print(f'create dir to save training log: {save_dir}')

	for e in tqdm(range(num_epoch)):
		epoch_loss = []
		for b, (X, y) in enumerate(train_loader):
			y_pred = model(X)
			loss = loss_fn(y_pred, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss.append(loss.item())

		epoch_loss = np.array(epoch_loss).mean()
		epoch_valid_results = test(valid_loader, model, metrics, loss_fn).tolist()
		log_train(e, model, epoch_loss, epoch_valid_results, log_dir=save_dir)


def log_train(epoch: int, model: nn.Module, loss, valid_results: list, log_dir: str):
	valid_results  = [loss] + valid_results
	with open(os.path.join(log_dir, f'epoch_{epoch}_metrics.csv'), mode='a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([epoch] + valid_results)

		print(f"Epoch {epoch} | loss: {loss: .4f}, metrics: {valid_results: .4f}")

	torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch}_checkpoint.pth'))


@torch.no_grad()
def test(
		test_loader: DataLoader,
		model: nn.Module,
		metrics: list,
		loss_fn = None
) -> np.ndarray:
	results = []
	for b, (X, y) in enumerate(test_loader):
		y_pred = model(X)
		batch_results = [metrics_fn(y, y_pred) for metrics_fn in metrics]

		if loss_fn is not None:
			loss = loss_fn(y_pred, y).item()
			batch_results = [loss] + batch_results

		results.append(batch_results)
	return np.array(results).mean(axis=0)
