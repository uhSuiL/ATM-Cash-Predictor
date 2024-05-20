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
		self.h0_fc = nn.Linear(input_dim, hidden_dim) if train_h0 else nn.Identity()

	def forward(self, time_series: torch.Tensor, h0: torch.Tensor = None):
		h0 = self.h0_fc(h0)
		normed_time_series = self.normalizer.normalize(time_series)

		output, hn = self.gru(normed_time_series, h0)

		output = self.fc(output)
		output = self.normalizer.denormalize(output)
		return output
