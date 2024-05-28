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


class MovingAverage(nn.Module):
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
		])  # (batch_size, num_features, num_steps)

		X = X.permute(0, 2, 1)  # (batch_size, num_features, num_steps)
		return X
