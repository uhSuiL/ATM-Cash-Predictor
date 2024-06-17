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
	name = 'SimpleGRU'

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


class TimeGRU(nn.Module):
	name = 'TimeGRU'

	def __init__(self,
				 input_dim: int, hidden_dim: int, output_dim: int, num_layer: int = 1, train_h0: bool = True,
				 time_embed: nn.Module = None):
		super().__init__()
		self.time_embed = nn.Identity() if time_embed is None else time_embed

		self.gru = layer.GRUPro(input_dim, hidden_dim, output_dim, num_layer, train_h0)
		self.normalizer = layer.Normalizer()

	def forward(self, X, time_embedding, h0):
		time_embedding = self.time_embed(time_embedding)
		X = self.normalizer.normalize(X)

		X = torch.concat([
			X,
			time_embedding
		], dim=-1)

		pred = self.gru(X, h0)
		pred = self.normalizer.denormalize(pred)
		return torch.squeeze(pred)


class Time2VecGRU(nn.Module):
	name = 'GRU-T'

	def __init__(self,
				 input_dim: int, hidden_dim: int, output_dim: int, pred_embed_dim: int, time_embed_dim: int,
				 num_layer: int = 1, train_h0: bool = True, t2v_act_fn = None):
		super().__init__()

		self.normalizer = layer.Normalizer()
		self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
		self.h0_fc = nn.Linear(hidden_dim, hidden_dim) if train_h0 else nn.Identity()

		self.t2v = layer.Time2Vec(time_embed_dim, t2v_act_fn)
		self.fc = nn.Linear(hidden_dim + time_embed_dim, output_dim)

	def forward(self, X, time_ticks, h0):
		if X.dim() == 2:  # set `batch_size=1`
			X, time_ticks = X.unsqueeze(dim=0), time_ticks.unsqueeze(dim=0)

		X = self.normalizer.normalize(X)
		h0 = self.h0_fc(h0)

		output, _ = self.gru(X, h0)  # (batch_size, num_steps, hidden_dim)
		time_embed = self.t2v(time_ticks)  # (batch_size, num_steps, time_embed_dim)
		preds = torch.concat([
			output,
			time_embed,
		], dim=-1)   # (batch_size, num_steps, output_dim)
		pred = self.fc(preds[:, -1:, :])

		pred = self.normalizer.denormalize(pred)
		return torch.squeeze(pred, dim=0).squeeze()


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
	name = 'DLinear'

	def __init__(self,
				 is_individual: bool, num_series: int, num_steps: int, num_pred_steps: int,
				 moving_avg: nn.Module = None,
				 num_exo_t_vars: int = 0, num_exo_s_vars: int = 0, keep_num_pred_dim: bool = False):
		super().__init__()
		self.num_pred_steps = num_pred_steps
		self.keep_num_pred_dim = keep_num_pred_dim

		self.normalizer = layer.Normalizer()
		self.d_linear = layer.DLinear(
			is_individual, num_series, num_steps, num_pred_steps,
			moving_avg,
			num_exo_t_vars, num_exo_s_vars)

	def forward(self, time_series: torch.Tensor, exo_t_vars: torch.Tensor = None, exo_s_vars: torch.Tensor = None):
		normed_time_series = self.normalizer.normalize(time_series)
		pred = self.d_linear(normed_time_series, exo_t_vars, exo_s_vars)
		pred = self.normalizer.denormalize(pred)
		return pred.squeeze(dim=1) if (self.num_pred_steps == 1) and (not self.keep_num_pred_dim) else pred


class Time2VecDLinear(nn.Module):
	"""Deprecated"""
	def __init__(self,
				 is_individual: bool, num_series: int, num_steps: int, num_pred_steps: int,
				 trend_embed_dim: int, season_embed_dim: int,
				 moving_avg: nn.Module = None):
		super().__init__()
		self.num_pred_steps = num_pred_steps

		self.normalizer = layer.Normalizer()
		self.trend_embed = layer.Time2Vec(trend_embed_dim, num_series, keep_dim_series=True)
		self.season_embed = layer.Time2Vec(season_embed_dim, num_series, keep_dim_series=True)
		self.d_linear = layer.DLinear(
			is_individual, num_series, num_steps, num_pred_steps,
			moving_avg,
			trend_embed_dim, season_embed_dim)

	def forward(self, time_series: torch.Tensor, time_ticks: torch.Tensor):
		"""
		:param time_series: ([batch_size], num_steps, num_series])
		:param time_ticks: ([batch_size], num_steps, 1)
		"""
		normed_time_series = self.normalizer.normalize(time_series)
		trend_embedding = self.trend_embed(time_ticks)
		season_embedding = self.season_embed(time_ticks)
		pred = self.d_linear(normed_time_series, trend_embedding, season_embedding)
		pred = self.normalizer.denormalize(pred)
		return pred.squeeze(dim=1) if self.num_pred_steps == 1 else pred


class Time2VecDLinear_2(nn.Module):
	name = 'DLinear-T'

	def __init__(self,
				 is_individual: bool, num_series: int, num_steps: int, num_pred_steps: int,
				 time_embed_dim: int,
				 num_exo_t_vars: int = 0, num_exo_s_vars: int = 0,
				 moving_avg: nn.Module = None,
				 keep_num_pred_dim: bool = False):
		super().__init__()
		self.num_pred_steps = num_pred_steps
		self.num_series = num_series
		self.keep_num_pred_dim = keep_num_pred_dim

		self.normalizer = layer.Normalizer()

		self.time_embed = layer.Time2Vec(time_embed_dim, num_series, keep_dim_series=True)
		self.time_proj1 = nn.Linear(time_embed_dim, 1)
		self.time_proj2 = nn.Linear(num_steps, num_pred_steps)

		# self.fc = layer.MLP([1 + time_embed_dim] + fc_hidden_dims + [num_pred_steps])
		self.d_linear = layer.DLinear(
			is_individual, num_series, num_steps, num_pred_steps,
			moving_avg,
			num_exo_t_vars, num_exo_s_vars)

	def forward(self, time_series: torch.Tensor, time_ticks: torch.Tensor, e_t = None, e_s = None):
		"""
		:param time_series: ([batch_size], num_steps, num_series])
		:param time_ticks: ([batch_size], num_steps, 1)
		:param e_t: ([batch_size], num_steps, num_exo_t_vars)
		:param e_s: ([batch_size], num_steps, num_exo_s_vars)
		"""
		normed_time_series = self.normalizer.normalize(time_series)

		time_embedding = self.time_embed(time_ticks)  # (batch_size, num_steps, num_series, embed_dim)
		time_embedding = self.time_proj1(time_embedding) \
			.squeeze(dim=-1).transpose(-1, -2)  # (batch_size, num_series, num_steps)
		time_embedding = self.time_proj2(time_embedding) \
			.transpose(-1, -2)  # (batch_size, num_pred_steps, num_series)

		pred = self.d_linear(normed_time_series, e_t, e_s)  # (batch_size, num_pred_steps, num_series)
		pred += time_embedding
		pred = self.normalizer.denormalize(pred)
		return pred.squeeze(dim=1) if (self.num_pred_steps == 1) and (not self.keep_num_pred_dim) else pred


class StrategicDLinear(nn.Module):
	name = 'DLinear-Strategy'

	def __init__(self, d_linear: DLinear, strategy: layer.Strategy):
		super().__init__()
		self.d_linear = d_linear
		self.strategy = strategy

	def forward(self, X, e_t, e_s, T_next, other_vars = None):
		pred_demand = self.d_linear(X, e_t, e_s)
		X_t = X[:, -1:, :]
		return self.strategy(pred_demand, X_t, T_next, other_vars)


class StrategicTime2VecDLinear(nn.Module):
	name = 'DLinear-T_Strategy'

	def __init__(self, t2v_d_linear: Time2VecDLinear_2, strategy: layer.Strategy, strategy_time_embed: int = None):
		super().__init__()
		self.t2v_d_linear = t2v_d_linear
		self.strategy = strategy

		# self.is_time_for_strategy = strategy_time_embed
		# if strategy_time_embed is not None:
		# 	self.t2v_strategy = layer.Time2Vec(strategy_time_embed, t2v_d_linear.num_series)

	def forward(self, X, ticks, T_next, e_t, e_s, other_vars):
		if X.dim() == 2:
			X = X.unsqueeze(dim=0)

		pred_demand = self.t2v_d_linear(X, ticks, e_t, e_s)
		# if self.is_time_for_strategy:
		# 	time_embed = self.t2v_d_linear(ticks)
		# 	other_vars = torch.concat([time_embed, other_vars], dim=-1)

		X_t = X[:, -1:, :]
		return self.strategy(pred_demand, X_t, T_next, other_vars)
