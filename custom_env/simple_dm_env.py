import dm_env
from dm_env import specs
import matplotlib.pyplot as plt

import numpy as np
import torch


class SimpleDMEnv(dm_env.Environment):
	metadata = {'render.modes': ['human', "rgb"]}

	def __init__(self, max_step, stochastic=True, limit=None):
		super(SimpleDMEnv, self).__init__()
		self.max_step = max_step
		self.stochastic = stochastic
		self.limit = limit

	def action_spec(self):
		return specs.BoundedArray(minimum=-1, maximum=1, shape=(2,), dtype=np.float32, name='action',)

	def observation_spec(self):
		return specs.BoundedArray(minimum=-self.limit, maximum=self.limit, shape=(2,), dtype=np.float32, name='observation',)

	def step(self, action):
		"""
		No reward for this env
		"""
		action = action.flatten()
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
		reward = 0
		observation = self.agent_pos.copy()
		discount = 1.0
		self.agent_traj.append(observation)
		if done:
			return dm_env.TimeStep(
				dm_env.StepType.LAST, np.array([reward]), np.array([discount]), np.array([observation]))
		else:
			return dm_env.TimeStep(dm_env.StepType.MID, np.array([reward]), np.array([1.0]), np.array([observation]))

	def reset(self):
		# self.agent_pos = np.random.uniform(size=2).astype(np.float32) * 2 * LIMIT - LIMIT
		self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
		self.step_count = 0
		self.agent_traj = []
		return dm_env.restart(np.array([self.agent_pos]))

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

	def plot_prediction_net(self, agent, cfg, step=0, device="cuda", anti=False, SHOW=False):
		NUM = 51  # LIMIT+1
		assert agent.domain == "toy"
		LIMIT = cfg.env.toy.limit
		x = np.linspace(-LIMIT, LIMIT, num=NUM)

		x_list = []
		y_list = []
		xlabel_list = []
		ylabel_list = []

		for i in range(NUM):
			for j in range(NUM):
				test_obs = torch.tensor([x[i], x[j]], device=device, dtype=torch.float32)
				if anti:
					predicted_z = torch.softmax(agent.anti_diayn(None, torch.tensor(test_obs, device=device)).
												reshape(cfg.agent.skill_channel, -1),
												dim=-1)
				else:
					predicted_z = torch.softmax(agent.diayn(None, torch.tensor(test_obs, device=device)).
												reshape(cfg.agent.skill_channel, -1),
												dim=-1)
				x_list.append(x[i])
				y_list.append(x[j])
				xlabel_list.append(predicted_z[0].argmax().item())
				ylabel_list.append(predicted_z[1].argmax().item())

		x_list = np.array(x_list)
		y_list = np.array(y_list)
		xlabel_list = np.array(xlabel_list)
		ylabel_list = np.array(ylabel_list)

		plt.clf()
		# plt.xlim(-LIMIT, LIMIT)
		# plt.ylim(-LIMIT, LIMIT)

		apd = f"_step_{step}"
		if anti:
			apd += "_anti"

		for c in np.unique(xlabel_list):
			plt.scatter(x_list[xlabel_list == c], y_list[xlabel_list == c], label=c)
		plt.legend()
		plt.savefig("dim1" + apd)
		if SHOW:
			plt.show()
		else:
			plt.clf()

		for c in np.unique(ylabel_list):
			plt.scatter(x_list[ylabel_list == c], y_list[ylabel_list == c], label=c)
		plt.legend()
		plt.savefig("dim2" + apd)
		if SHOW:
			plt.show()
		else:
			plt.clf()

		# Plot combined:
		for c in range(cfg.agent.skill_dim ** cfg.agent.skill_channel):
			x_cord = c % cfg.agent.skill_dim
			y_cord = c // cfg.agent.skill_dim
			merged_idx = (xlabel_list == x_cord) * (ylabel_list == y_cord)
			plt.scatter(x_list[merged_idx],
						y_list[merged_idx], label=str(x_cord) + str(y_cord))
		plt.legend()
		plt.savefig("combined" + apd)
		if SHOW:
			plt.show()



