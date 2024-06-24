from train_simple_gru import valid
from data import SlidingWinDataset
from model import util, StrategicGRU

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import l1_loss


import pandas as pd
import numpy as np


def train_strategic_gru(
		train_data: pd.DataFrame,
		SLIDING_WIN = 10,
		BATCH_SIZE = 4,
		SHUFFLE = True,
		NUM_EPOCH = 400,
		train_h0: bool = True,
		save_dir: str = './log/StrategicGRU',
):
	valid_data = train_data.iloc[-20:, :]
	train_data = train_data.iloc[:-20, :]

	INPUT_DIM = train_data.shape[-1]

	train_set = SlidingWinDataset(train_data, SLIDING_WIN)
	valid_set = SlidingWinDataset(valid_data, SLIDING_WIN)

	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
	valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False)

	HIDDEN_DIM = 10
	model = StrategicGRU(
		input_dim=INPUT_DIM,
		hidden_dim=HIDDEN_DIM,
		output_dim=INPUT_DIM,
		train_h0=True,
		mlp_hidden_params=[16, 8]
	)
	optimizer = torch.optim.Adam(model.parameters())
	loss_fn = nn.MSELoss()
	metrics = [l1_loss]

	save_dir += f'(norm, train_h0={train_h0}, win={SLIDING_WIN}, batch_size={BATCH_SIZE})'

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
	train_data = pd.read_csv('./data/13series_time_stacked_cash/train.csv').drop(['日期'], axis=1).astype(float)
	train_strategic_gru(
		train_data,
		save_dir='./log/Strategic(MLP)GRU'
	)
