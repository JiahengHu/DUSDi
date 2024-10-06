"""
Wrapper around moma 2d
Defines:
- Reward
- Additional necessary states
- New observation space
"""

import gym
from gym import spaces
import matplotlib.pyplot as plt

import numpy as np
import torch
import math
from .moma_2d_gym_env import MoMa2DGymEnv


def to_one_hot(idx, num_classes):
	one_hot = np.zeros(num_classes, dtype=int)
	if idx < 0:
		return one_hot
	one_hot[idx] = 1
	return one_hot


class MoMa2DGymDSEnv(MoMa2DGymEnv):
	"""Custom Environment that follows gym interface"""

	def __init__(self, max_step, show_empty, version):
		self.version = version
		super(MoMa2DGymDSEnv, self).__init__(max_step, show_empty)
		obs_low = np.array([0]*17 + [-self.view_limit, 0], dtype=np.float32)
		obs_high = np.array([1]*12 + [self.limit]*4 + [self.gripper_limit, self.view_limit, 1000], dtype=np.float32)
		self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(19,), dtype=np.float32)
		self.max_step = max_step

	def get_reward(self):
		# We just have a fixed goal for now: if the agent achieves, gets a reward of one
		self.goal = [0, 0, 0]

		# Change the goal base depending on time
		if self.step_count < 250:
			self.goal[0] = 0
			self.goal[2] = 0
		elif self.step_count < 500:
			self.goal[0] = 1
			self.goal[2] = 1
		elif self.step_count < 750:
			self.goal[0] = 2
			self.goal[2] = 2
		else:
			self.goal[0] = 3
			self.goal[2] = 3

		# Reward is calculated based on the low level steps
		reward = 0
		# Goal: base, arm, view,
		if self.base_item[self.goal[0]] == 1:
			reward += 1
		if self.view_item[self.goal[2]] == 1:
			reward += 1
		if self.version == "lim":
			if self.arm_item[self.goal[1]] == 1:
				reward *= 1
			else:
				reward = 0
		else:
			assert self.version == "nolim"

		return reward

	# Defines additional states needed for the upper policy
	def get_additional_states(self):
		return [self.step_count / self.max_step].copy()
