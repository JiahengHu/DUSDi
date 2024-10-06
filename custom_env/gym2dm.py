import dm_env
from dm_env import specs
import gym
from gym import spaces
import numpy as np


def gym_timestep_to_dm_env_timestep(gym_timestep, n_envs, discount=1.0):
	if gym_timestep is None:
		return None
	else:
		observation, reward, done, info = gym_timestep
		if n_envs == 1:
			# We manually vectorize the environment here
			observation = np.array([observation], dtype=np.float32)
			reward = np.array([reward], dtype=np.float32)
			done = np.array([done], dtype=bool)
			info = np.array([info], dtype=object)
		else:
			observation = np.array(observation, dtype=np.float32)
		if done.any():
			assert done.all()
			new_discount = discount * np.ones([n_envs, 1], dtype=np.float32)
			return dm_env.TimeStep(
				dm_env.StepType.LAST, reward, new_discount, observation), info
		else:
			new_discount = np.ones([n_envs, 1], dtype=np.float32)
			return dm_env.TimeStep(dm_env.StepType.MID, reward, new_discount, observation), info


def gym_reset_to_dm_env_reset(gym_reset, n_env):
	if gym_reset is None:
		return None
	else:
		if n_env == 1:
			gym_reset = np.array([gym_reset], dtype=np.float32)
		else:
			gym_reset = np.array(gym_reset, dtype=np.float32)
		return dm_env.restart(gym_reset)


def gym_space_to_dm_specs(gym_space, name):
	if isinstance(gym_space, spaces.Box):
		return specs.BoundedArray(minimum=gym_space.low, maximum=gym_space.high,
								  shape=gym_space.shape, dtype=gym_space.dtype, name=name)
	elif isinstance(gym_space, spaces.Discrete):
		return specs.DiscreteArray(num_values=gym_space.n, name=name)
	else:
		raise NotImplementedError


# Convert a gym environment to a dm_env environment.
class Gym2DMWrapper(dm_env.Environment):
	def __init__(self, env, n_env):
		self._env = env
		self.n_env = n_env
		self._action_spec = gym_space_to_dm_specs(env.action_space, name="action")
		self._obs_spec = gym_space_to_dm_specs(env.observation_space, name="observation")
		self.info = None

	def reset(self):
		return gym_reset_to_dm_env_reset(self._env.reset(), self.n_env)

	def step(self, action):
		time_step, info = gym_timestep_to_dm_env_timestep(self._env.step(action), self.n_env)
		self.info = info
		return time_step

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._action_spec

	def __getattr__(self, name):
		return getattr(self._env, name)
