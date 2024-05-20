import pandas as pd
import torch
from torch.utils.data import Dataset


class SlidingWinDataset(Dataset):
	def __init__(self, time_series: pd.DataFrame, sliding_win: int):
		time_series =  time_series.to_numpy()

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
