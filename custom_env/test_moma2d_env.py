import numpy as np
from moma_2d_gym_env import MoMa2DGymEnv

env = MoMa2DGymEnv(1000)
print(env.observation_space)
print(env.action_space)

s = env.reset()

for i in range(1000):
    a = env.action_space.sample()
    s, r, done, info = env.step(a)

    print(f"current action: {a}")
    print(f"next observation: {s}")
    # print(ts)
    env.render(mode="human", block=True)

