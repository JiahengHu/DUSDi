"""
this wrapper will be used for all environments: serve as a wrapper for hierarchical agent
"""

import gym
from gym import spaces
import numpy as np
import torch
from gym.wrappers import RescaleAction

def to_one_hot(idx, num_classes):
	one_hot = np.zeros(num_classes, dtype=int)
	if idx < 0:
		return one_hot
	one_hot[idx] = 1
	return one_hot

class HierarchicalDiscreteEnv(gym.Env):
	'''
	This env is for multi diayn
	'''
	def __init__(self, env, skill_channel, skill_dim, low_level_steps, device, low_actor=None, vis=False):
		super(HierarchicalDiscreteEnv, self).__init__()

		# This is necessary (since we always apply this in training)
		self._env = RescaleAction(env, min_action=-1, max_action=1)
		self.action_space = spaces.MultiDiscrete([skill_dim] * skill_channel)
		self.observation_space = self._env.observation_space
		self.low_level_actor = low_actor
		self.low_level_steps = low_level_steps  # 50
		self.device = device
		self.skill_channel = skill_channel
		self.skill_dim = skill_dim
		self.vis = vis

	# This function will extract the additional states from the env to generate observations for high-level policy
	def get_full_state(self, obs):
		full_obs = np.concatenate([obs, self._env.get_additional_states()])
		return full_obs

	def reset(self):
		self.last_observation = self._env.reset()
		return self.get_full_state(self.last_observation.copy())

	def step(self, meta_action):
		reward = 0

		for i in range(self.low_level_steps):
			if self.low_level_actor is None:
				print("low level actor is None")
				action = self._env.action_space.sample()
			else:
				with torch.no_grad():
					obs = torch.as_tensor(self.last_observation, device=self.device, dtype=torch.float32).flatten()
					inputs = [obs]

					skill = np.zeros((self.skill_channel, self.skill_dim), dtype=np.float32)
					skill[range(self.skill_channel), meta_action] = 1.0

					value = torch.as_tensor(skill, device=self.device).flatten()
					inputs.append(value)

					inpt = torch.cat(inputs, dim=-1)
					# We are assumming using SAC
					dist = self.low_level_actor(inpt, 0.2)
					action = dist.mean.cpu().numpy()

			observation, r, done, info = self._env.step(action)
			if self.vis:
				self.render("human")
			reward += r
			if done:
				break
			self.last_observation = observation
		reward /= self.low_level_steps

		return self.get_full_state(observation), reward, done, info

	def render(self, mode="human"):
		return self._env.render(mode)

	def __getattr__(self, name):
		return getattr(self._env, name)


class HierarchicalContinuousEnv(HierarchicalDiscreteEnv):
	'''
	This is for cic
	'''
	def __init__(self, env, skill_dim, low_level_steps, device, low_actor=None, vis=False):

		# This is necessary (since we always apply this in training)
		self._env = RescaleAction(env, min_action=-1, max_action=1)
		# This is derived from the CIC skill space
		self.action_space = spaces.Box(low=np.zeros(skill_dim), high=np.ones(skill_dim), dtype=np.float32)
		self.observation_space = self._env.observation_space
		self.low_level_actor = low_actor
		self.low_level_steps = low_level_steps  # 50
		self.device = device
		self.skill_dim = skill_dim
		self.vis = vis

	def step(self, meta_action):
		reward = 0

		for i in range(self.low_level_steps):
			if self.low_level_actor is None:
				print("WARNING: low level actor is None")
				action = self._env.action_space.sample()
			else:
				with torch.no_grad():
					obs = torch.as_tensor(self.last_observation, device=self.device, dtype=torch.float32).flatten()
					inputs = [obs]

					value = torch.as_tensor(meta_action, device=self.device).flatten()
					inputs.append(value)

					inpt = torch.cat(inputs, dim=-1)
					# We don't really need a distribution here...
					dist = self.low_level_actor(inpt, 0.2)
					action = dist.mean.cpu().numpy()

			observation, r, done, info = self._env.step(action)
			if self.vis:
				self.render("human")
			reward += r
			if done:
				break
			self.last_observation = observation
		reward /= self.low_level_steps

		return self.get_full_state(observation), reward, done, info


class HierarchicalDiaynEnv(HierarchicalDiscreteEnv):
	'''
	This is for diayn
	'''
	def __init__(self, env, skill_dim, low_level_steps, device, low_actor=None, vis=False):

		# This is necessary (since we always apply this in training)
		self._env = RescaleAction(env, min_action=-1, max_action=1)
		# This is derived from the CIC skill space
		self.action_space = spaces.Discrete(skill_dim)
		self.observation_space = self._env.observation_space
		self.low_level_actor = low_actor
		self.low_level_steps = low_level_steps  # 50
		self.device = device
		self.skill_dim = skill_dim
		self.vis = vis

	def step(self, meta_action):
		reward = 0

		for i in range(self.low_level_steps):
			if self.low_level_actor is None:
				print("WARNING: low level actor is None")
				action = self._env.action_space.sample()
			else:
				with torch.no_grad():
					obs = torch.as_tensor(self.last_observation, device=self.device, dtype=torch.float32).flatten()
					inputs = [obs]

					skill = np.zeros(self.skill_dim, dtype=np.float32)
					skill[meta_action] = 1.0

					value = torch.as_tensor(skill, device=self.device).flatten()
					inputs.append(value)

					inpt = torch.cat(inputs, dim=-1)
					# We don't really need a distribution here...
					dist = self.low_level_actor(inpt, 0.2)
					action = dist.mean.cpu().numpy()

			observation, r, done, info = self._env.step(action)
			if self.vis:
				self.render("human")
			reward += r
			if done:
				break
			self.last_observation = observation
		reward /= self.low_level_steps

		return self.get_full_state(observation), reward, done, info


class FlatEnvWrapper(gym.Env):
	"""
	This is to wrap env for exploration methods
	Defines reward aggregation and observation concatenation
	"""
	def __init__(self, env, low_level_steps, vis=False, ):
		super(FlatEnvWrapper, self).__init__()

		# This is necessary (since we always apply this in training)
		self._env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		self.low_level_steps = low_level_steps  # 50
		self.vis = vis

	# This function will extract the additional states from the env to generate observations for high-level policy
	def get_full_state(self, obs):
		full_obs = np.concatenate([obs, self._env.get_additional_states()])
		return full_obs

	def reset(self):
		observation = self._env.reset()
		self.cycle_reward = 0
		self.step_count = 0
		return self.get_full_state(observation)

	def step(self, action):
		self.step_count += 1

		observation, r, done, info = self._env.step(action)
		self.cycle_reward += r
		if self.step_count % self.low_level_steps == 0:
			reward = self.cycle_reward / self.low_level_steps
			self.cycle_reward = 0
		else:
			reward = 0

		return self.get_full_state(observation), reward, done, info

	def render(self, mode="human"):
		return self._env.render(mode)

	def __getattr__(self, name):
		return getattr(self._env, name)
