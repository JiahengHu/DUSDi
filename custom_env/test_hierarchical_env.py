import numpy as np
from moma_2d_downstream_env import MoMa2DGymDSEnv
from hierarchical_env_wrapper import HierarchicalDiscreteEnv

env = MoMa2DGymDSEnv(1000)

wrap = True
if wrap:
    env = HierarchicalDiscreteEnv(env, 3, 3, 50, "cpu")
    print(env.observation_space)
    print(env.action_space)

s = env.reset()

total_reward = 0
for i in range(1000):
    a = env.action_space.sample()
    observation, reward, done, _ = env.step(a)
    total_reward += reward

    if done:
        print(total_reward)
        env.save_traj("test")
        exit()
        env.reset()

    print(f"current action: {a}")
    print(f"next observation: {observation}")
    print(f"reward: {reward}")
    print(f"current step: {i}")
    env.render(mode="human", block=True)

