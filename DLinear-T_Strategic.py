# %%
from model import StrategicTime2VecDLinear, Time2VecDLinear_2, util, layer
from data import SlidingWinDataset

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np


def train(
		COUNTER_NUM,
		SEED,
		SLIDING_WIN = 10,
		BATCH_SIZE = 4,
		SHUFFLE = True,
		VALID_SIZE = 70,
):
	# ==============================
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	# ==============================
	train_data = pd.read_csv('./data/13series_time_stacked_cash/train.csv').drop(['日期'], axis=1).astype(float)
	train_data.columns = [int(col.split('_')[-1]) for col in train_data.columns]

	info = pd.read_csv('./data/Group2-ATM-Info.csv', index_col=0)
	info['在/离行'] = info['在/离行'].map({'在行': 1, '离行': 0})
	info['是否开启循环'] = info['是否开启循环'].map({'是': 1, '否': 0})

	train_data = pd.concat([
		train_data[COUNTER_NUM],
		pd.Series(train_data.reset_index().index),
		pd.DataFrame(
			info[info['柜员号'] == COUNTER_NUM].drop('柜员号', axis=1).to_numpy().repeat(repeats=train_data.shape[0],
																						 axis=0)
		),
	], axis=1)

	valid_data = train_data.iloc[-VALID_SIZE:]
	train_data = train_data.iloc[:-VALID_SIZE]

	NUM_SERIES = 1

	train_set = SlidingWinDataset(train_data, sliding_win=SLIDING_WIN)
	valid_set = SlidingWinDataset(valid_data, sliding_win=SLIDING_WIN)

	train_loader = DataLoader(train_set, batch_size=4, shuffle=SHUFFLE)
	valid_loader = DataLoader(valid_set, batch_size=4, shuffle=SHUFFLE)

	# moving_avg = layer.UltimateSmoother()
	moving_avg = layer.SimpleMovingAverage(win_length=5)
	# moving_avg = layer.WeightedMovingAverage(SLIDING_WIN, num_features=NUM_SERIES)

	model = StrategicTime2VecDLinear(
		t2v_d_linear=Time2VecDLinear_2(
			is_individual=True, num_series=NUM_SERIES, num_steps=SLIDING_WIN, num_pred_steps=1,
			time_embed_dim=7,
			moving_avg=moving_avg,
			keep_num_pred_dim=True
		),
		strategy=layer.Strategy(
			num_series=NUM_SERIES,
			time_embed_dim=2,
			# mlp_hidden_params=[4],
		)
	)

	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	loss_fn = nn.MSELoss()
	metrics = [nn.functional.l1_loss]
	SAVE_DIR = f'./log/{model.name}(norm,{moving_avg.name},win={SLIDING_WIN},batch_size={BATCH_SIZE},valid_size={VALID_SIZE})/{COUNTER_NUM}/{SEED}'
	NUM_EPOCH = 400

	@torch.no_grad()
	def valid(_test_loader) -> np.ndarray:
		# results = []
		for b, (X, y) in enumerate(_test_loader):
			O = X[:, :, 2:].unsqueeze(dim=-2)
			T = X[:, :, 1:2]
			T_next = T[:, -1:, :] + 1
			X = X[:, :, :1]
			y = y[:, :1]
			y_pred = model(X, e_t=None, e_s=None, ticks=T, T_next=T_next, other_vars=None).squeeze(dim=-2)

			batch_results = [metrics_fn(y, y_pred).item() for metrics_fn in metrics]

			if loss_fn is not None:
				loss = loss_fn(y_pred, y).item()
				batch_results = [loss] + batch_results

			return batch_results

	for e in range(0, NUM_EPOCH):
		epoch_loss = []
		for b, (X, y) in enumerate(train_loader):
			O = X[:, :, 2:].unsqueeze(dim=-2)
			T = X[:, :, 1:2]
			T_next = T[:, -1:, :] + 1
			X = X[:, :, :1]

			y = y[:, :1]

			y_pred = model(X, e_t=None, e_s=None, ticks=T, T_next=T_next, other_vars=None).squeeze(dim=-2)
			loss = loss_fn(y_pred, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss.append(loss.item())

		epoch_loss = np.array(epoch_loss).mean()
		epoch_valid_results = valid(valid_loader)
		util.log_train(e, model, epoch_loss, epoch_valid_results, save_dir=SAVE_DIR)


if __name__ == '__main__':
	seeds = range(19, 20)
	# counter_nums = [9012, 9003, 9049, 9025, 9053, 9077, 9207, 9164, 9008, 9039, 9200, 9472, 9490]
	# counter_nums = [9472, 9490]
	# counter_nums = [9049]
	counter_nums = [9200]
	for counter_num in counter_nums:
		for seed in seeds:
			print(f"Train {counter_num}, seed={seed}")
			train(
				COUNTER_NUM=counter_num,
				SEED=seed,
			)
