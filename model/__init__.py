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
	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layer: int = 1, h0: torch.tensor = None):
		super().__init__()

		self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
		self.normalizer = layer.Normalizer()
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.h0 = nn.Parameter(torch.zeros(num_layer, hidden_dim)) if h0 is None else h0

	def forward(self, time_series: torch.Tensor):
		if len(time_series.shape) == 3:
			batch_size = time_series.shape[0]
			a = self.h0.expand(self.gru.num_layers, batch_size, self.gru.hidden_size).contiguous()
			self.h0 = a

		normed_time_series = self.normalizer.normalize(time_series)
		output, hn = self.gru(normed_time_series, self.h0)

		output = self.fc(output)
		output = self.normalizer.denormalize(output)
		return output
