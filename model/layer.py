from math import sqrt, pi

import numpy as np

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
	def __init__(self, win_length: int):
		super().__init__()
		self.win_length = win_length
		self.moving_avg = nn.AvgPool1d(kernel_size=win_length, stride=1)

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
			X[:, :, :self.win_length - 1],
			self.moving_avg(X)
		], dim=-1)  # (batch_size, num_features, num_steps)

		X_smooth = X_smooth.permute(0, 2, 1)  # (batch_size, num_features, num_steps)
		return X_smooth


class UltimateSmoother(nn.Module):
	"""refer: https://www.mesasoftware.com/papers/UltimateSmoother.pdf"""
	def __init__(self, period: int | list, train_period: bool):
		super().__init__()
		self.period = torch.tensor(period, dtype=torch.float)  # (1,) | (num_features,)

		# if train_period:  # CAN'T USE !!!
		# 	self.period = nn.Parameter(self.period)

	def forward(self, X):  # (batch_size, num_steps, num_features)
		a1 = torch.exp(-sqrt(2) * pi / self.period)
		c3 = - a1 ** 2
		c2 = -2 * a1 * torch.cos(sqrt(2) * 180 / self.period)
		c1 = (1 + c2 - c3) / 4

		if X.dim() == 2:  # (num_steps, num_features)
			X = torch.unsqueeze(X, dim=0)  # (1, num_steps, num_features)

		X = X.permute(1, 0, 2)  # (num_steps, batch_size, num_features)

		X_smooth = X.clone()
		for t in range(X.shape[0])[3:]:
			X_smooth[t] = (
					(1 - c1) * X[t - 1]
					+ (2 * c1 - c2) * X[t - 2]
					- (c1 + c3) * X[t - 3]
					+ c2 * X_smooth[t - 1]
					+ c3 * X_smooth[t - 2]
			)

		X_smooth = X_smooth.permute(1, 0, 2)  # (batch_size, num_steps, num_features)
		return X_smooth


def ultimate_smooth(X: np.ndarray, period: int = 20, auto_fill: bool = False):
	# X shape: (T, )
	a1 = np.exp(-sqrt(2) * pi / period)
	c3 = -a1 ** 2
	c2 = -2 * a1 * np.cos(sqrt(2) * 180 / period)
	c1 = (1 + c2 - c3) / 4

	X_smooth = X.copy()
	for t in range(X.shape[0])[3:]:
		X_smooth[t] = (
				(1 - c1) * X[t - 1]
				+ (2 * c1 - c2) * X[t - 2]
				- (c1 + c3) * X[t - 3]
				+ c2 * X_smooth[t - 1]
				+ c3 * X_smooth[t - 2]
		)

	return X_smooth if auto_fill else X_smooth[3:]


# input_size, hidden_size, num_layers = 10, 20, 2
# batch_size = 5
# input = torch.randn(batch_size, 3, input_size)  # (batch_size, num_steps, input_size)
# rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
# h0 = torch.randn(num_layers, batch_size, 20)
# output, hn = rnn(input, h0)
class GRUPro(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layer: int = 1, train_h0: bool = True):
		super().__init__()

		self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.h0_fc = nn.Linear(hidden_dim, hidden_dim) if train_h0 else nn.Identity()

	def forward(self, time_series: torch.Tensor, h0: torch.tensor):
		h0 = self.h0_fc(h0)  # (num_layers, batch_size, hidden_dim)

		output, hn = self.gru(time_series, h0)

		# output the last pred
		if len(output.shape) == 3:  # (batch_size, num_steps, hidden_dim)
			output = torch.unsqueeze(output[:, -1, :], dim=1)
		elif len(output.shape) == 2:  # (num_steps, hidden_dim)
			output = torch.unsqueeze(output[-1, :], dim=0)
		else:
			raise RuntimeError(f"Shape Illegal: {output.shape}")

		pred = self.fc(output)
		# return torch.squeeze(pred)
		return pred


class Time2Vec(nn.Module):
	def __init__(self, embed_dim: int, activation_fn = None):
		super().__init__()

		self.f = torch.sin if activation_fn is None else activation_fn
		self.omg = nn.Parameter(torch.randn(1, embed_dim))
		self.phi = nn.Parameter(torch.randn(embed_dim))

	def forward(self, time_ticks: torch.Tensor):  # shape: ([batch_size], time_len, 1)
		if time_ticks.dim() == 2:
			time_ticks = torch.unsqueeze(time_ticks, dim=0)

		time_embed = time_ticks @ self.omg + self.phi  # (batch_size, time_len, embed_dim)
		time_embed = torch.concat([
			time_embed[:, :, :1],
			self.f(time_embed[:, :, 1:])
		], dim=-1)
		return time_embed  # .squeeze(dim=0)  # ([batch_size], time_len, embed_dim)


class DLinear(nn.Module):
	def __init__(self,
				 is_individual: bool, num_series: int, num_steps: int, num_pred_steps: int,
				 moving_avg: nn.Module = None):
		super().__init__()
		self.is_individual = is_individual

		init_w_shape = (num_series, num_pred_steps, num_steps) if is_individual else (num_pred_steps, num_steps)
		init_weights = torch.ones(*init_w_shape) / num_steps

		self.W_t = nn.Parameter(init_weights)
		self.b_t = nn.Parameter(torch.zeros(num_pred_steps, num_series))

		self.W_s = nn.Parameter(init_weights)
		self.b_s = nn.Parameter(torch.zeros(num_pred_steps, num_series))

		self.moving_avg = moving_avg

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
