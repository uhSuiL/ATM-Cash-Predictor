from model import util, SimpleGRU
from data import SlidingWinDataset

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import l1_loss

import pandas as pd
import numpy as np


@torch.no_grad()
def valid(valid_loader, model, hidden_dim, metrics, loss_fn = None) -> list:
	for b, (X, y) in enumerate(valid_loader):
		assert b < 1, "Only one batch is expected"

		y_pred = model(X, torch.zeros(1, X.shape[0], hidden_dim, dtype=torch.float))
		batch_results = [metrics_fn(y, y_pred).item() for metrics_fn in metrics]

		if loss_fn is not None:
			loss = loss_fn(y_pred, y).item()
			batch_results = [loss] + batch_results

		return batch_results


def train_simple_gru(
	SLIDING_WIN = 10,
	BATCH_SIZE = 8,
	SHUFFLE = True,
	NUM_EPOCH = 1000,
):
	train_data = pd.read_csv('./data/13series_time_stacked_cash/train.csv').drop(['日期'], axis=1).astype(float)

	valid_data = train_data.iloc[-20:, :]
	train_data = train_data.iloc[:-20, :]

	INPUT_DIM = train_data.shape[-1]

	train_set = SlidingWinDataset(train_data, SLIDING_WIN)
	valid_set = SlidingWinDataset(valid_data, SLIDING_WIN)

	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
	valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False)

	HIDDEN_DIM = 10
	model = SimpleGRU(
		input_dim=INPUT_DIM,
		hidden_dim=HIDDEN_DIM,
		output_dim=INPUT_DIM,
		train_h0=True
	)
	optimizer = torch.optim.Adam(model.parameters())
	loss_fn = nn.MSELoss()
	metrics = [l1_loss]

	save_dir = f'./log/SimpleGRU(norm, h0 trainable, win={SLIDING_WIN}, batch_size={BATCH_SIZE})/'

	for e in range(0, NUM_EPOCH):
		epoch_loss = []
		for b, (X, y) in enumerate(train_loader):
			y_pred = model(X, torch.zeros(1, X.shape[0], HIDDEN_DIM, dtype=torch.float))

			loss = loss_fn(y_pred, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss.append(loss.item())

		epoch_loss = np.array(epoch_loss).mean()
		epoch_valid_results = valid(
			valid_loader,
			model,
			HIDDEN_DIM,
			metrics,
			loss_fn
		)
		util.log_train(e, model, epoch_loss, epoch_valid_results, save_dir=save_dir)


if __name__ == '__main__':
	train_simple_gru()
