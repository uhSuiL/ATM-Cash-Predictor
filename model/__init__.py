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
