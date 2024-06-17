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
	data.index = data.index.rename("Epoch")

	if metrics_names is None:
		metrics_names = [f'metrics_{i}' for i in range(data.shape[-1] - 1)]
	data.columns = ['train_loss'] + metrics_names

	print(f"start drawing figure, data shape: {data.shape}")

	data.plot(figsize=(16, 8) if figsize is None else figsize, title="Training Log")

	if v_lines is not None and len(v_lines) > 0:
		for x in v_lines:
			plt.axvline(x=x, linestyle='--')

	return data


def load_checkpoint(model: nn.Module, save_dir: str, date: str, epoch: int):
	f_path = os.path.join(save_dir, date, f'epoch_{epoch}_checkpoint.pth')
	state_dict = torch.load(f_path)
	model.load_state_dict(state_dict)
	return model


def visualize_trains(save_dir, counter, col, metric_name, date=None, start=None, end=None, is_show=True):
	# save_dir: dir to model name: model_name(config)/counter/seed/date/train.csv
	# train_log.csv: train_loss, valid_loss, valid_metrics
	save_dir = os.path.join(save_dir, str(counter))
	seeds = os.listdir(save_dir)
	logs = []
	for seed in seeds:
		if date is None:
			last_date = os.listdir(os.path.join(save_dir, seed))[-1]
			log_path = os.path.join(save_dir, seed, last_date, 'train_log.csv')
		else:
			log_path = os.path.join(save_dir, seed, date, 'train_log.csv')

		log = pd.read_csv(log_path, index_col=0, header=None).iloc[start: end, col].rename(metric_name)
		if log.isna().any().any():
			print(seed)
		if log.shape[0] != 300:
			...
		logs.append(log)

	df = pd.concat(logs, axis=1)
	arr = df.to_numpy()# (num_steps, num_train)

	mean = np.mean(arr, axis=-1)
	std = np.std(arr, axis=-1)
	x = df.index

	plot_band(x, mean, mean - std, mean + std, metric_name)
	if is_show:
		plt.show()
	return mean, std
	# return mean[-1]


def plot_band(x, main, lower, upper, title):
	# TODO: SET DEFFERENT COLOR
	plt.title(title)
	plt.plot(x, main, color='orange')
	plt.fill_between(x, lower, upper, alpha=0.5, color='skyblue')

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
