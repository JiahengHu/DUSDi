import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from functorch import combine_state_for_ensemble
from functorch import vmap
from torch.nn.utils.rnn import pad_sequence

from .partition_utils import get_env_factorization
from .networks.value_head import FactoredValueHead, SeparateValueHead
from .params import separate_sac_reward

# Returns zeros no matter what's the prediction
class DummyCritic(nn.Module):
	def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
		super().__init__()
		return None

	def forward(self, obs, action):
		batch_size = obs.shape[0]
		q1 = torch.zeros(batch_size, 1).to(obs.device)
		q2 = torch.zeros(batch_size, 1).to(obs.device)
		return q1, q2


class Critic(nn.Module):
	def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
		super().__init__()

		self.obs_type = obs_type

		if obs_type == 'pixels':
			# for pixels actions will be added after trunk
			self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
									   nn.LayerNorm(feature_dim), nn.Tanh())
			trunk_dim = feature_dim + action_dim
		else:
			# for states actions come in the beginning
			self.trunk = nn.Sequential(
				nn.Linear(obs_dim + action_dim, hidden_dim),
				nn.LayerNorm(hidden_dim), nn.Tanh())
			trunk_dim = hidden_dim

		def make_q():
			q_layers = []
			q_layers += [
				nn.Linear(trunk_dim, hidden_dim),
				nn.ReLU(inplace=True)
			]
			if obs_type == 'pixels':
				q_layers += [
					nn.Linear(hidden_dim, hidden_dim),
					nn.ReLU(inplace=True)
				]
			q_layers += [nn.Linear(hidden_dim, 1)]
			return nn.Sequential(*q_layers)

		self.Q1 = make_q()
		self.Q2 = make_q()

		self.apply(utils.weight_init)

	def forward(self, obs, action):
		inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
															   dim=-1)
		h = self.trunk(inpt)
		h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

		q1 = self.Q1(h)
		q2 = self.Q2(h)

		return q1, q2


class BranchCritic(nn.Module):
	"""
	Basically a regular critic, except that we branch out towards the end
	"""


	def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, use_sac, skill_channels, ext_r_dim):
		super().__init__()


		if separate_sac_reward and use_sac:
			reward_channels = skill_channels + 1
		else:
			reward_channels = skill_channels

		reward_channels += ext_r_dim


		# for states actions come in the beginning
		self.trunk = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim),
			nn.LayerNorm(hidden_dim), nn.Tanh())
		trunk_dim = hidden_dim

		def make_q():
			q_layers = []
			q_layers += [
				nn.Linear(trunk_dim, hidden_dim),
				nn.ReLU(inplace=True)
			]
			q_layers += [nn.Linear(hidden_dim, reward_channels)]
			return nn.Sequential(*q_layers)

		self.Q1 = make_q()
		self.Q2 = make_q()

		self.apply(utils.weight_init)


	def forward(self, obs, action):
		inpt = torch.cat([obs, action],dim=-1)
		h = self.trunk(inpt)

		q1 = self.Q1(h)
		q2 = self.Q2(h)

		Q = torch.stack([q1, q2], -1)

		return Q

class SepCritic(nn.Module):
	"""
	Completely separate critic network, speed up with the factored value head
	"""
	def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, skill_channels, use_sac, ext_r_dim):
		super().__init__()

		if separate_sac_reward and use_sac:
			reward_channels = skill_channels + 1
		else:
			reward_channels = skill_channels

		reward_channels += ext_r_dim

		self.value_head = SeparateValueHead(obs_dim + action_dim, reward_channels,
											num_layers=2, hidden_size=512, Q_range=None)


	def forward(self, obs, action):
		x = torch.cat([obs, action], dim=-1)
		values = self.value_head(x)
		return values



# For pre-processing input modalities
class FeatureExtractor(nn.Module):
	def __init__(
		self,
		skill_dim,
		skill_channel,
		embed_size,
		domain,
		extra_num_layers=0,
		extra_hidden_size=64,
	):
		"""
		Maps state factors and skill factors into tokens
		"""
		super().__init__()
		self.obs_partition, self.skill_partition, self.action_partition = get_env_factorization(
			domain, skill_dim, skill_channel)
		self.skill_dim = skill_dim
		self.skill_channel = skill_channel

		# We can also try more coarse partitions - maybe later
		self.num_tokens = len(self.obs_partition) + len(self.skill_partition) + len(self.action_partition)

		extra_encoders_list = []

		def generate_proprio_mlp_fn(factor_size):
			assert factor_size > 0  # we indeed have extra information
			if extra_num_layers > 0:
				layers = [nn.Linear(factor_size, extra_hidden_size)]
				for i in range(1, extra_num_layers):
					layers += [
						nn.Linear(extra_hidden_size, extra_hidden_size),
						nn.ReLU(inplace=True),
					]
				layers += [nn.Linear(extra_hidden_size, embed_size)]
			else:
				layers = [nn.Linear(factor_size, embed_size)]

			proprio_mlp = nn.Sequential(*layers)
			extra_encoders_list.append(proprio_mlp)

		all_tokens = self.skill_partition + self.obs_partition + self.action_partition
		self.max_dim = max(all_tokens)

		for _ in all_tokens:
			generate_proprio_mlp_fn(self.max_dim)

		fmodel, params, buffers = combine_state_for_ensemble(extra_encoders_list)
		self.v_params = [nn.Parameter(p) for p in params]
		self.v_buffers = [nn.Buffer(b) for b in buffers]

		for i, param in enumerate(self.v_params):
			self.register_parameter('encoder_param_' + str(i), param)

		for i, buffer in enumerate(self.v_buffers):
			self.register_buffer('encoder_buffer_' + str(i), buffer)

		self.vmap_model = vmap(fmodel)

	def forward(self, obs_list):
		"""
		obs_list: a list of [B, k], where k is specified by the obs_partition
		map above to a latent vector of shape (B, num_factors, H)
		"""

		minibatches = pad_sequence([obs.T for obs in obs_list], batch_first=True)  # (num_factors, max_dim, B)
		minibatches = torch.swapaxes(minibatches, 1, 2)  # (num_factors, B, max_dim)

		x = self.vmap_model(self.v_params, self.v_buffers, minibatches)  # (num_factors, B, E)
		x = torch.swapaxes(x, 0, 1)  # (B, num_factors, E)

		return x

class SimpleAttnCritic(nn.Module):
	"""
	Using a learnable weighting / max of the embeddings
	"""
	def __init__(self, skill_dim, skill_channels, domain, agg):
		super().__init__()

		embed_size = 256
		hidden_size = 512
		self.agg = agg

		self.feature_encoder = FeatureExtractor(
			domain=domain,
			skill_dim=skill_dim,
			skill_channel=skill_channels,
			embed_size=embed_size,
		)

		if separate_sac_reward:
			reward_channels = skill_channels + 1
		else:
			reward_channels = skill_channels

		attn_logit = nn.Parameter(torch.randn([reward_channels, self.feature_encoder.num_tokens]))
		self.register_parameter("attn_logit", attn_logit)
		self.value_head = FactoredValueHead(embed_size, reward_channels, num_layers=2, hidden_size=hidden_size)

	def forward(self, obs_parts, skill_parts, action_parts,
				skill_obs_dependency, skill_action_dependency, skill_skill_dependency):
		x = self.feature_encoder(skill_parts + obs_parts + action_parts)  # (B, num_tokens, E)
		x = self.attention_forward(x, self.attn_logit)  # (B, num_skills, E)
		values = self.value_head(x)
		return values

	def attention_forward(self, x, attn_logits):
		attn = torch.softmax(attn_logits, dim=-1)  # (num_skills, num_tokens)

		if self.agg == "max":
			# For taking the max
			attn = torch.unsqueeze(attn, -1)
			x = torch.unsqueeze(x, 1)  # after the batch dimension
			features = x * attn
			features, _ = features.max(dim=2)
		elif self.agg == "avg":
			# For taking the weighted average
			features = torch.matmul(x.transpose(-2, -1), attn.T).transpose(-2, -1)
		return features


class StateMaskCritic(nn.Module):
	"""
	Directly perform masking in the input space
	So essentially, concatenation instead of max / weighted average
	"""
	def __init__(self, obs_dim, action_dim, skill_dim, skill_channels, domain, device,
				 weighted, use_sac, topk_gating, attn_balancing, ext_r_dim, Q_range):
		super().__init__()

		self.obs_partition, self.skill_partition, self.action_partition = get_env_factorization(
			domain, skill_dim, skill_channels)

		self.factor_list = self.obs_partition + self.skill_partition + self.action_partition
		self.num_tokens = len(self.factor_list)
		self.factor_list = torch.tensor(self.factor_list, requires_grad=False, device=device)
		self.topk_gating = topk_gating
		self.bal_loss = None
		self.attn_balancing = attn_balancing

		if separate_sac_reward and use_sac:
			reward_channels = skill_channels + 1
		else:
			reward_channels = skill_channels

		reward_channels += ext_r_dim

		if weighted:
			attn_logit = nn.Parameter(torch.randn([reward_channels, self.num_tokens]))
		else:
			attn_logit = nn.Parameter(torch.ones([reward_channels, self.num_tokens]), requires_grad=False)
		self.register_parameter("attn_logit", attn_logit)

		self.value_head = FactoredValueHead(obs_dim + action_dim, reward_channels,
											num_layers=2, hidden_size=512, Q_range=Q_range)

	def forward(self, obs, action):
		x = torch.cat([obs, action], dim=-1)
		x = self.attention_forward(x, self.attn_logit) # This takes a lot of time...
		values = self.value_head(x)
		return values

	def attention_forward(self, x, attn_logits):
		if self.topk_gating:
			attn = self.topk_gating_function(attn_logits, k=10) # TODO: topk setting is pretty hacky
		else:
			attn = torch.softmax(attn_logits, dim=-1)  # (num_skills, num_tokens)

		# Calculate the attention balancing loss
		if self.attn_balancing:
			total_prob = attn.sum(dim=0)
			self.bal_loss = F.mse_loss(total_prob, torch.ones_like(total_prob) * attn_logits.shape[0] / self.num_tokens)

		attn = attn.repeat_interleave(self.factor_list, dim=-1)  # (num_skills, obs_dim)
		x = x.unsqueeze(1)
		features = x * attn  # (B, num_skills, obs_dim)
		return features

	# Unlike moe, this is data independent gating
	def topk_gating_function(self, attn_logits, k):
		_, topk_idx = torch.topk(attn_logits, k, dim=-1)
		mask = torch.zeros_like(attn_logits, dtype=torch.bool)
		mask[torch.arange(attn_logits.shape[0]).unsqueeze(-1), topk_idx] = 1

		straight_through = True
		if not straight_through:
			with torch.no_grad():
				attn_logits[~mask] = -torch.inf
			attn = torch.softmax(attn_logits, dim=-1) # (num_skills, num_tokens)
		else:
			# straight-through trick
			attn_softmax = torch.softmax(attn_logits, dim=-1)
			attn = mask.float() + attn_softmax - attn_softmax.detach()
		return attn
