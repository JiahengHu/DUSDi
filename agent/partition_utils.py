import torch
import numpy as np


# these values will be initialized upon agent creation, in utils.py
SIMP_PAR = None
USE_IMG = None


def get_env_obs_act_dim(domain, env_config):
	if domain == "particle":
		obs_dim = 70
		if env_config.particle.simplify_action_space:
			action_dim = 20
		else:
			action_dim = 50
	else:
		obs_part, _, action_part = get_env_factorization(domain, 0, 0)
		obs_dim = sum(obs_part)
		action_dim = sum(action_part)

	return obs_dim, action_dim

# legacy function, not actually useful (i.e. the factorization can be arbitrary)
def get_env_factorization(domain, skill_dim, skill_channel):
	if "moma2d" in domain:
		obs_partition = [4, 4, 4, 2, 3, 1]  # base, arm, view, base, arm, view
		action_partition = [2, 3, 1]  # base, arm, view
	elif domain == "particle":
		N = skill_channel
		if USE_IMG:
			obs_partition = [1]*N + [2]*N + [2]*N  # No longer have velocity
		else:
			obs_partition = [1]*N + [4]*N + [2]*N  # lm1-3, pos_vel1-3, lm1-3
		if SIMP_PAR:
			action_partition = [2]*N
		else:
			action_partition = [5]*N

	elif domain == "igibson":
		obs_partition = [3, 4, 3, 8, 7, 7, 2, 5, 1]
		action_partition = [2, 2, 3, 3, 1]
	elif domain == "wipe":
		obs_partition = [96]
		action_partition = [17]
	else:
		# For other domain, this is not implemented yet
		raise NotImplementedError

	skill_partition = [skill_dim] * skill_channel
	return obs_partition, skill_partition, action_partition



###############################
# Diayn related specification #
###############################

# specifies how the diayn vector should be partitioned
def get_domain_stats(domain, env_config):
	FULLBOX = env_config.igibson.fullbox
	sep_obj = env_config.igibson.sep_obj
	N = env_config.particle.N

	config = env_config[domain]

	if domain == "toy":
		diayn_dim = 2
		state_partition_points = [0, 1, 2]
	elif domain == "igibson":
		assert FULLBOX
		assert not sep_obj
		diayn_dim = 10
		state_partition_points = [0, 3, 7, 10]
	elif domain == "moma2d":
		diayn_dim = 12
		state_partition_points = [0, 4, 8, 12]
	elif domain == "particle":
		diayn_dim = N * 1
		state_partition_points = list(range(0, diayn_dim+1))
	else:
		raise NotImplementedError

	assert state_partition_points[-1] == diayn_dim
	assert state_partition_points[0] == 0

	return diayn_dim, state_partition_points


# specifies how the diayn vector should be extracted from observation
def observation_filter(obs, domain, env_config):
	if obs is None:
		return None

	config = env_config[domain]

	# We always process in batch
	if len(obs.shape) == 1:
		obs = obs.unsqueeze(0)

	if domain == "toy":
		return obs
	elif domain == "igibson":
		idx = np.array(range(10))
		return obs[:, idx]
	elif domain == "wipe":
		idx = np.array(range(*config.diayn_idx))
		return obs[:, idx]
	elif domain == "moma2d":
		idx = np.array(range(12))
		return obs[:, idx]
	elif domain == "particle":
		idx = np.array(range(env_config.particle.N))
		return obs[:, idx]
	else:
		print("Domain {} not supported".format(domain))
		raise NotImplementedError


# This function is used for partitioning input into different parts
def obtain_partitions(obs, skill, action, domain, skill_dim, skill_channel):
	obs_partition, skill_partition, action_partition = get_env_factorization(domain, skill_dim, skill_channel)
	# Next, partition both the obs and the skill
	skill_list = torch.split(skill, skill_partition, dim=-1)
	obs_list = torch.split(obs, obs_partition, dim=-1)
	action_list = torch.split(action, action_partition, dim=-1)

	return obs_list, skill_list, action_list
