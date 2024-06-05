from math import sqrt, pi

import torch
from torch import nn


def MLP(layer_dims: list, Activation: nn.Module = None):
	assert len(layer_dims) > 1, len(layer_dims)

	Activation = nn.LeakyReLU if Activation is None else Activation
	layers = []
	for i in range(len(layer_dims) - 2):
		layers += [nn.Linear(layer_dims[i], layer_dims[i + 1]), Activation()]
	layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
	return nn.Sequential(*layers)


class Normalizer:
	def __init__(self, dim=-2, epsilon=1e-5):
		self.mean = 0
		self.std = 1

		self.dim = dim
		self.epsilon = epsilon

	@torch.no_grad()
	def normalize(self, time_series: torch.tensor):
		self.mean = time_series.mean(dim=self.dim, keepdim=True)
		self.std = time_series.std(dim=self.dim, keepdim=True)
		return (time_series - self.mean) / (self.std + self.epsilon)

	# @torch.no_grad()
	def denormalize(self, normed_time_series: torch.tensor):
		return self.mean + (self.std + self.epsilon) * normed_time_series


class SimpleMovingAverage(nn.Module):
	def __init__(self, win_length: int, stride: int = 1):
		super().__init__()
		self.win_length = win_length
		self.moving_avg = nn.AvgPool1d(kernel_size=win_length, stride=stride)

	def forward(self, X: torch.Tensor):  # (batch_size, num_steps, num_features)
		if X.dim() == 2:  # (num_steps, num_features)
			X = torch.unsqueeze(X, dim=0)  # (1, num_steps, num_features)

		X = X.permute(0, 2, 1)  # (batch_size, num_features, num_steps)

		# Keep sequence length
		X = torch.concat([
			X[:, :, : self.win_length - 1],
			self.moving_avg(X)  # (batch_size, num_features, num_steps - (win_length - 1))
		], dim=-1)  # (batch_size, num_features, num_steps)

		X = X.permute(0, 2, 1)  # (batch_size, num_features, num_steps)
		return X


class WeightedMovingAverage(nn.Module):
	def __init__(self, win_length: int, num_features: int):
		super().__init__()
		self.win_length = win_length
		self.moving_avg = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=win_length)

	def forward(self, X: torch.Tensor):  # (batch_size, num_steps, num_features)
		if X.dim() == 2:  # (num_steps, num_features)
			X = torch.unsqueeze(X, dim=0)  # (1, num_steps, num_features)

		X = X.permute(0, 2, 1)  # (batch_size, num_features, num_steps)
		X_smooth = torch.concat([
			X[:, :, :self.win_length],
			self.moving_avg(X)
		], dim=-1)  # (batch_size, num_features, num_steps)

		X_smooth = X_smooth.permute(0, 2, 1)  # (batch_size, num_features, num_steps)
		return X_smooth


class UltimateSmoother(nn.Module):
	"""refer: https://www.mesasoftware.com/papers/UltimateSmoother.pdf"""
	def __init__(self, period: int | list, train_period: bool):
		super().__init__()
		self.period = torch.tensor(period, dtype=torch.float)  # (1,) | (num_features,)

		if train_period:
			self.period = nn.Parameter(self.period)

	def forward(self, X):  # (batch_size, num_steps, num_features)
		a1 = torch.exp(-sqrt(2) * pi / self.period)
		c3 = - a1 ** 2
		c2 = -2 * a1 * torch.cos(sqrt(2) * 180 / self.period)
		c1 = (1 + c2 - c3) / 4

		if X.dim() == 2:  # (num_steps, num_features)
			X = torch.unsqueeze(X, dim=0)  # (1, num_steps, num_features)

		X = X.permute(1, 0, 2)  # (num_steps, batch_size, num_features)

		X_smooth = X.clone()
		for t in range(X.shape[0])[2:]:
			X_smooth[t] = (
					(1 - c1) * X[t]
					+ (2 * c1 - c2) * X[t - 1]
					- (c1 + c3) * X[t - 2]
					+ c2 * X_smooth[t - 1]
					+ c3 * X_smooth[t - 2]
			)

		X_smooth = X_smooth.permute(1, 0, 2)  # (batch_size, num_steps, num_features)
		return X_smooth


class DLinear(nn.Module):
	def __init__(self,
				 is_individual: bool, num_series: int, num_steps: int, num_pred_steps: int,
				 ma_win_len: int, ma_stride: int = 1):
		super().__init__()
		self.is_individual = is_individual

		init_w_shape = (num_series, num_pred_steps, num_steps) if is_individual else (num_pred_steps, num_steps)
		init_weights = torch.ones(*init_w_shape) / num_steps

		self.W_t = nn.Parameter(init_weights)
		self.b_t = nn.Parameter(torch.zeros(num_pred_steps, num_series))

		self.W_s = nn.Parameter(init_weights)
		self.b_s = nn.Parameter(torch.zeros(num_pred_steps, num_series))

		self.moving_avg = SimpleMovingAverage(ma_win_len, ma_stride)

	def forward(self, X: torch.Tensor):
		"""ATTENTION: MAKE SURE DIMENSION INCLUDES `num_series`"""
		if X.dim() == 2:  # (num_steps, num_series) -- add `batch_size = 1`
			X = X.unsqueeze(dim=0)

		X_t = self.moving_avg(X)  # (batch_size, num_steps, num_series)
		X_s = X - X_t  # (batch_size, num_steps, num_series)

		X_t = X_t.permute(0, 2, 1).unsqueeze(dim=-1)  # (batch_size, num_series, num_steps, 1)
		X_s = X_s.permute(0, 2, 1).unsqueeze(dim=-1)  # (batch_size, num_series, num_steps, 1)

		# (num_series, num_pred_steps, num_steps) x (batch_size, num_series, num_steps, 1)
		H_t = self.W_t @ X_t  # (batch_size, num_series, num_pred_steps, 1)
		H_s = self.W_s @ X_s  # (batch_size, num_series, num_pred_steps, 1)

		H_t = H_t.squeeze(dim=-1).permute(0, 2, 1)  # (batch_size, num_pred_steps, num_series)
		H_s = H_s.squeeze(dim=-1).permute(0, 2, 1)  # (batch_size, num_pred_steps, num_series)

		X_hat = (H_t + self.b_t) + (H_s + self.b_s)  # (batch_size, num_pred_steps, num_series)
		return X_hat
