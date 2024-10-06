import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


class SimpleGymEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self, max_step, stochastic=True, limit=None):
		super(SimpleGymEnv, self).__init__()
		self.max_step = max_step
		self.stochastic = stochastic
		self.limit = limit
		self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-self.limit, high=self.limit, shape=(2,), dtype=np.float32)

	def reset(self):
		self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
		self.step_count = 0
		self.agent_traj = []
		return self.agent_pos

	def step(self, action):
		"""
		No reward for this env
		"""
		self.agent_pos += action

		if self.stochastic:
			NOISE_LEVEL = 1
			self.agent_pos += np.random.normal(scale=NOISE_LEVEL, size=self.agent_pos.shape[0])

		for i in range(self.agent_pos.shape[0]):
			if self.agent_pos[i] > self.limit:
				self.agent_pos[i] = self.limit
			elif self.agent_pos[i] < -self.limit:
				self.agent_pos[i] = -self.limit
		self.step_count += 1
		done = self.step_count >= self.max_step
		reward = 0.0
		observation = self.agent_pos.copy()
		discount = 1.0
		self.agent_traj.append(observation)

		return observation, reward, done, {}

	def save_traj(self, fn):
		plt.clf()
		plt.xlim(-self.limit, self.limit)
		plt.ylim(-self.limit, self.limit)
		agent_traj = np.array(self.agent_traj)
		plt.plot(agent_traj[:, 0], agent_traj[:, 1])
		plt.title(self.step_count)
		plt.savefig(fn)

	def render(self, mode='rgb', close=False):
		plt.clf()
		plt.xlim(-self.limit, self.limit)
		plt.ylim(-self.limit, self.limit)
		plt.scatter(self.agent_pos[0], self.agent_pos[1])
		plt.title(self.step_count)
		if mode == 'human':
			plt.show(block=False)
			plt.pause(0.0001)
			return None
		elif mode == 'rgb':
			fig = plt.gcf()
			fig.canvas.draw()
			rgba_buf = fig.canvas.buffer_rgba()
			w, h = fig.canvas.get_width_height()
			rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3].copy()
			return rgba_arr
		else:
			raise NotImplementedError
