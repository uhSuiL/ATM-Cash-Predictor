import os
import csv
from datetime import datetime

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
		log_train(e, model, epoch_loss, epoch_valid_results, save_dir=save_dir)


def log_train(epoch: int, model: nn.Module, train_loss, valid_results: list, save_dir: str):
	# save_domain/model_name/datetime/
	save_dir = os.path.join(save_dir, now.strftime('%Y-%m-%d-%H-%M'))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		print(f'create dir to save training log: {save_dir}')

	with open(os.path.join(save_dir, f'train_log.csv'), mode='a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([epoch, train_loss] + valid_results)

		print(f"Epoch {epoch} | train loss: {train_loss: .4f} | valid metrics: {valid_results}")

	torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}_checkpoint.pth'))


def visualize_train(log_dir: str, metrics_names: list = None, *, v_lines: list = None, figsize: tuple = None) -> pd.DataFrame:
	sns.set()

	data = pd.read_csv(os.path.join(log_dir, 'train_log.csv'), header=None)

	data = data.set_index(0, drop=True)  # Epoch num as index
	if metrics_names is None:
		metrics_names = [f'metrics_{i}' for i in range(data.shape[-1] - 1)]
	data.columns = ['train_loss'] + metrics_names

	data.plot(figsize=(16, 8) if figsize is None else figsize)

	if v_lines is not None and len(v_lines) > 0:
		for x in v_lines:
			plt.axvline(x=x, linestyle='--')

	return data


def load_checkpoint(model: nn.Module, save_dir: str, date: str, epoch: int):
	f_path = os.path.join(save_dir, date, f'epoch_{epoch}_checkpoint.pth')
	state_dict = torch.load(f_path)
	model.load_state_dict(state_dict)


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
