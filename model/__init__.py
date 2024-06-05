from . import layer

import torch
from torch import nn


# input_size, hidden_size, num_layers = 10, 20, 2
# batch_size = 5
# input = torch.randn(batch_size, 3, input_size)  # (batch_size, num_steps, input_size)
#
# rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
# h0 = torch.randn(num_layers, batch_size, 20)
# output, hn = rnn(input, h0)
class SimpleGRU(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layer: int = 1, train_h0: bool = True):
		super().__init__()

		self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
		self.normalizer = layer.Normalizer()
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.h0_fc = nn.Linear(hidden_dim, hidden_dim) if train_h0 else nn.Identity()

	def forward(self, time_series: torch.Tensor, h0: torch.tensor):
		h0 = self.h0_fc(h0)  # (num_layer, batch_size, hidden_dim)
		normed_time_series = self.normalizer.normalize(time_series)

		output, hn = self.gru(normed_time_series, h0)

		# output the last pred
		if len(output.shape) == 3:  # (batch_size, num_steps, hidden_dim)
			output = torch.unsqueeze(output[:, -1, :], dim=1)
		elif len(output.shape) == 2:  # (num_steps, hidden_dim)
			output = torch.unsqueeze(output[-1, :], dim=0)
		else:
			raise RuntimeError(f"Shape Illegal: {output.shape}")

		pred = self.fc(output)
		pred = self.normalizer.denormalize(pred)
		return torch.squeeze(pred)


class StrategicGRU(nn.Module):
	def __init__(
			self,
			input_dim: int, hidden_dim: int, output_dim: int, num_layer: int = 1, train_h0: bool = True,
			mlp_hidden_params: list = None,
	):
		super().__init__()
		self.strategy_fc = nn.Linear(input_dim * 2, input_dim) if mlp_hidden_params is None \
			else layer.MLP([input_dim * 2] + mlp_hidden_params + [input_dim])

		self.simple_gru = SimpleGRU(input_dim, hidden_dim, output_dim, num_layer, train_h0)
		# self.strategy_fc = nn.Linear(input_dim * 2, input_dim)
		# self.strategy_mlp = layer.MLP([input_dim * 2] + mlp_hidden_params + [input_dim])
		self.relu = nn.ReLU()

	def forward(self, time_series: torch.Tensor, h0: torch.tensor):
		if time_series.dim() == 2:
			time_series = time_series.unsqueeze(dim=0)  # (batch_size=1, num_step, feature_dim)

		demand = self.simple_gru(time_series, h0)  # (batch_size, input_dim)
		replenish = self.strategy_fc(
			torch.concat((demand, time_series[:, -1, :]), dim=-1)  # (batch_size, input_dim x 2)
		)  # (batch_size, input_dim)
		pred = self.relu(time_series[:, -1, :] + replenish - demand)
		return pred  # (batch_size, input_dim)


class DLinear(nn.Module):
	def __init__(self, is_individual: bool, num_series: int, num_steps: int, num_pred_steps: int, ma_win_len: int):
		super().__init__()
		self.num_pred_steps = num_pred_steps

		self.normalizer = layer.Normalizer()
		self.d_linear = layer.DLinear(is_individual, num_series, num_steps, num_pred_steps, ma_win_len)

	def forward(self, time_series: torch.Tensor):
		normed_time_series = self.normalizer.normalize(time_series)
		pred = self.d_linear(normed_time_series)
		pred = self.normalizer.denormalize(pred)
		return pred.squeeze(dim=1) if self.num_pred_steps == 1 else pred
