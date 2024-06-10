import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWinDataset(Dataset):
	def __init__(self, time_series: pd.DataFrame, sliding_win: int):
		time_series = time_series.to_numpy(dtype=np.float32)

		# sliding_win -> time_series length
		# predict s[t] based on s[t-1]~s[t-1-win]
		self.windows = [
			torch.tensor(time_series[t - sliding_win: t])
			for t in range(sliding_win, time_series.shape[0])
		]
		self.labels = [
			torch.tensor(time_series[t])
			for t in range(sliding_win, time_series.shape[0])
		]

	def __getitem__(self, win_id):
		return self.windows[win_id], self.labels[win_id]

	def __len__(self):
		assert len(self.windows) == len(self.labels), f'{len(self.windows), len(self.labels)}'
		return len(self.windows)


def get_period_dummies(time_len: int, period_list: list[int], phase: int = 0) -> pd.DataFrame:
	period_dummies = []
	for p in period_list:
		ticks = [(t + phase) % p for t in range(time_len)]
		dummies = pd.get_dummies(ticks, prefix=f'period_{p}')
		assert dummies.shape[-1] == p, f'{dummies.shape} != {p}'
		period_dummies.append(dummies)
	return pd.concat(period_dummies, axis=1)
