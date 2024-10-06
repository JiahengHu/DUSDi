import torch
import torch.nn as nn
import utils


class Actor(nn.Module):
	def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, sac, log_std_bounds, domain):
		super().__init__()

		self.sac = sac
		feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

		self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		policy_layers = []
		policy_layers += [
			nn.Linear(feature_dim, hidden_dim),
			nn.ReLU(inplace=True)
		]
		# add additional hidden layer for pixels
		if obs_type == 'pixels':
			policy_layers += [
				nn.Linear(hidden_dim, hidden_dim),
				nn.ReLU(inplace=True)
			]

		if self.sac:
			policy_layers += [nn.Linear(hidden_dim, 2 * action_dim)]
		else:
			policy_layers += [nn.Linear(hidden_dim, action_dim)]

		self.policy = nn.Sequential(*policy_layers)
		self.log_std_bounds = log_std_bounds

		self.domain = domain

		self.apply(utils.weight_init)

	def forward(self, obs, std):
		h = self.trunk(obs)

		if self.sac:
			mu, log_std = self.policy(h).chunk(2, dim=-1)

			# constrain log_std inside [log_std_min, log_std_max]
			log_std = torch.tanh(log_std)
			log_std_min, log_std_max = self.log_std_bounds
			log_std = log_std_min + 0.5 * (log_std + 1) * (log_std_max - log_std_min)
			std = log_std.exp()
			dist = utils.SquashedNormal(mu, std)
		else:
			mu = self.policy(h)
			mu = torch.tanh(mu)
			std = torch.ones_like(mu) * std
			dist = utils.TruncatedNormal(mu, std)
		return dist


class SkillActor(nn.Module):
	"""
	This network is very similar to the normal Actor.
	However, it first processes observations and latent skills separately
	- rarely used
	"""
	def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, sac, log_std_bounds, skill_dim):
		super().__init__()

		self.sac = sac
		self.skill_dim = skill_dim
		feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

		obs_dim = obs_dim - skill_dim  # skill dim should be the total skill dim (i.e. skill * channel)
		self.obs_dim = obs_dim
		self.obs_trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
										nn.LayerNorm(feature_dim), nn.Tanh())

		self.skill_trunk = nn.Sequential(nn.Linear(skill_dim, feature_dim),
										nn.LayerNorm(feature_dim), nn.Tanh())

		policy_layers = []
		policy_layers += [
			nn.Linear(feature_dim * 2, hidden_dim),
			nn.ReLU(inplace=True)
		]

		# add additional hidden layer for pixels
		if obs_type == 'pixels':
			policy_layers += [
				nn.Linear(hidden_dim, hidden_dim),
				nn.ReLU(inplace=True)
			]

		if self.sac:
			policy_layers += [nn.Linear(hidden_dim, 2 * action_dim)]
		else:
			policy_layers += [nn.Linear(hidden_dim, action_dim)]

		self.policy = nn.Sequential(*policy_layers)
		self.log_std_bounds = log_std_bounds

		self.apply(utils.weight_init)

	def forward(self, obs, std):
		obs, skill = torch.split(obs, [self.obs_dim, self.skill_dim], dim=-1)
		obs_h = self.obs_trunk(obs)
		skill_h = self.skill_trunk(skill)
		h = torch.concat([obs_h, skill_h], dim=-1)

		if self.sac:
			mu, log_std = self.policy(h).chunk(2, dim=-1)

			# constrain log_std inside [log_std_min, log_std_max]
			log_std = torch.tanh(log_std)
			log_std_min, log_std_max = self.log_std_bounds
			log_std = log_std_min + 0.5 * (log_std + 1) * (log_std_max - log_std_min)
			std = log_std.exp()

			dist = utils.SquashedNormal(mu, std)
		else:
			mu = self.policy(h)
			mu = torch.tanh(mu)
			std = torch.ones_like(mu) * std
			dist = utils.TruncatedNormal(mu, std)
		return dist


class MCPActor(nn.Module):
	"""
	- this network should take in the observation,
	- parse the observation and feed into sub-policies different z's
			# 1. Create a nn moduleList of skills
			# 2. a weight
	- Right now, we have a state encoder, which branches out into each action module,
		in the mean time, each action module takes in an additional z vector
	# Maybe the state should also include previous actions?
	"""
	def __init__(self, obs_type, obs_skill_dim, action_dim, feature_dim, hidden_dim, sac, log_std_bounds,
				 skill_channel, skill_dim, use_gate):
		super().__init__()

		self.sac = sac
		self.use_gate = use_gate
		feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

		obs_dim = self.obs_dim = obs_skill_dim - skill_dim * skill_channel
		self.skill_dim = skill_dim
		self.skill_channel = skill_channel
		self.action_dim = action_dim

		self.primitive_state_encoder = nn.Sequential(
			nn.Linear(obs_dim, feature_dim),
			nn.ReLU(),
			nn.Linear(feature_dim, feature_dim),
			nn.ReLU(),
		)

		if self.sac:
			action_layer_size = self.action_dim * 2
		else:
			# Even for ddpg, we still output the std for primitives
			action_layer_size = self.action_dim * 2

		self.gate = nn.Sequential(
			nn.Linear(obs_skill_dim, feature_dim),
			nn.ReLU(),
			nn.Linear(feature_dim, skill_channel),
			nn.Sigmoid()
		)

		self.skill_encoders = nn.ModuleList(
			[nn.Linear(skill_dim, feature_dim) for _ in range(skill_channel)])
		self.primitives = nn.ModuleList(
			[nn.Sequential(nn.Linear(feature_dim + feature_dim, feature_dim), nn.ReLU(),
							nn.Linear(feature_dim, action_layer_size)) for _ in range(skill_channel)])

		self.log_std_bounds = log_std_bounds
		self.apply(utils.weight_init)

	def forward_primitives(self, obs_features, skill, idx):
		skill_ft = self.skill_encoders[idx](skill)
		input = torch.concat([obs_features, skill_ft], dim=-1)
		out = self.primitives[idx](input)
		mu, log_std = torch.split(out, self.action_dim, -1)
		mu = torch.tanh(mu)

		# constrain log_std inside [log_std_min, log_std_max]
		log_std = torch.tanh(log_std)
		log_std_min, log_std_max = self.log_std_bounds
		log_std = log_std_min + 0.5 * (log_std + 1) * (log_std_max - log_std_min)

		sigma = torch.ones_like(mu) * log_std.exp()
		return mu, sigma

	def forward_weights(self, obs_skill, device):
		if self.use_gate:
			weights = self.gate(obs_skill)
			# Expand on the action dimension level
			weights = weights.unsqueeze(dim=-2)
			return weights
		else:
			return torch.ones(self.skill_channel, device=device) * 0.5

	def forward(self, obs_skill, std):
		if len(obs_skill.shape) == 3:
			assert obs_skill.shape[0] == 1
			obs_skill = obs_skill.squeeze(0)
		obs, skills = torch.split(obs_skill, [self.obs_dim, self.skill_dim * self.skill_channel], dim=-1)
		skill_list = torch.split(skills, self.skill_dim, dim=-1)

		prim_embed = self.primitive_state_encoder(obs)

		outs = [self.forward_primitives(prim_embed, skill_list[i], i) for i in range(self.skill_channel)]
		mus, sigmas = zip(*outs)

		mus = torch.stack(mus, -1)
		sigmas = torch.stack(sigmas, -1)
		weights = self.forward_weights(obs_skill, sigmas.device)

		denom = (weights / sigmas).sum(-1)
		unnorm_mu = (weights / sigmas * mus).sum(-1)

		mean = unnorm_mu / denom

		scale_tril = 1 / denom

		# For calculating sum
		self.mus = mus
		self.sigmas = sigmas
		self.cmb_sigma = scale_tril
		self.gate_weights = weights
		self.unnorm_mu = unnorm_mu
		self.act_mean = mean

		if self.sac:
			dist = utils.SquashedNormal(mean, scale_tril)
		else:
			std = torch.ones_like(mean) * std
			dist = utils.TruncatedNormal(mean, std)

		return dist


class SeparateSkillActor(nn.Module):
	"""
	Use two separate actor networks, one for each action parts
	"""
	def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, sac,
				 log_std_bounds, skill_channel, skill_dim):
		super().__init__()

		self.sac = sac
		feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

		obs_dim = self.obs_dim = obs_dim - skill_dim * skill_channel
		self.skill_dim = skill_dim
		self.skill_channel = skill_channel
		self.action_dim = action_dim

		self.obs_trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
										nn.LayerNorm(feature_dim), nn.Tanh())

		self.skill_encoders = nn.ModuleList(
			[nn.Linear(skill_dim, feature_dim) for _ in range(skill_channel)])

		self.primitives = nn.ModuleList(
			[nn.Sequential(nn.Linear(feature_dim + feature_dim, feature_dim), nn.ReLU(),
						   nn.Linear(feature_dim, 1)) for _ in range(2)])

		if self.sac:
			raise NotImplementedError
		self.log_std_bounds = log_std_bounds

		self.apply(utils.weight_init)

	def forward(self, obs, std):
		obs, skill = torch.split(obs, [self.obs_dim, self.skill_dim * self.skill_channel], dim=-1)

		skill_list = torch.split(skill, self.skill_dim, dim=-1)

		prim_embed = self.obs_trunk(obs)

		outs = [self.forward_primitives(prim_embed, skill_list[i], i) for i in range(self.skill_channel)]

		mu = torch.concat(outs, dim=-1)
		mu = torch.tanh(mu)
		dist = utils.TruncatedNormal(mu, std)
		return dist

	def forward_primitives(self, obs_features, skill, idx):
		skill_ft = self.skill_encoders[idx](skill)
		input = torch.concat([obs_features, skill_ft], dim=-1)
		out = self.primitives[idx](input)
		return out



