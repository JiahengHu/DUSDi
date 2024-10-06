"""
Modified based on pretrain.py
This file trains downstream tasks
"""

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
from env_helper import make_ds_envs

import hydra
import numpy as np
import torch
from dm_env import specs

import wrapper
import utils
from logger import Logger
from hydra.utils import get_original_cwd, to_absolute_path

torch.backends.cudnn.benchmark = True
from utils import make_agent
from agent.partition_utils import get_env_obs_act_dim

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback
import wandb


def get_causal_matrix(domain, ds_task):
    assert domain == "particle"
    assert "poison" in ds_task

    if ds_task == "poison_s":
        agent_list = [1]
    elif ds_task == "poison_m":
        agent_list = [0, 2, 4, 6, 9]
    elif ds_task == "poison_l":
        agent_list = [0, 1, 2, 3, 4, 6, 7, 9]

    # N_reward * N_action
    causal_matrix = torch.zeros([len(agent_list), 10], dtype=torch.float32)
    for i in range(len(agent_list)):
        causal_matrix[i, agent_list[i]] = 1

    return causal_matrix, len(agent_list)


# If we want to have additional access to training
class CustomCallback(WandbCallback):
    def on_rollout_end(self) -> None:
        return super().on_rollout_end()


class Workspace:
    def __init__(self, cfg):
        cfg = self.ds_params_overwrite(cfg)

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device("cuda:{}".format(cfg.cuda_id))
        cfg.agent.nstep = 1  # just a place holder

        low_obs, low_act = get_env_obs_act_dim(cfg.domain, cfg.env)
        low_obs_spec = np.zeros(low_obs)
        low_act_spec = np.zeros(low_act)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                low_obs_spec,
                                low_act_spec,
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg,
                                cfg.agent)
        self.agent.init_critic = False

        if self.cfg.agent.name in ["dusdi_diayn"]:
            from omegaconf import OmegaConf, open_dict
            with open_dict(cfg):
                cfg.agent.skill_channel = self.agent.diayn_skill_channel
                cfg.agent.diayn_skill_channel = self.agent.diayn_skill_channel

        self.agent = self.load_agent(self.agent)

        actor = self.agent.actor

        make_envs_func = make_ds_envs(cfg, actor, self.device)

        # Function callback to create environments
        def make_env(rank: int, seed: int = 0):
            def _init():
                env = make_envs_func()
                env.seed(seed + rank)
                return env

            set_random_seed(seed + rank)
            return _init

        if self.cfg.n_env == 1 and not self.cfg.parallel_wrapper:
            self.train_env = make_envs_func()
        else:
            self.train_env = VecMonitor(SubprocVecEnv([make_env(i, self.cfg.seed) for i in range(cfg.n_env)]))
        self.eval_env = make_envs_func(vis=True)

    def ds_params_overwrite(self, cfg):
        """
        For convenience, we overwrite all domain-specific parameters here
        These are all caused by specific skill learning settings
        """

        agent_name = cfg.agent.name
        domain = cfg.domain
        cfg.env.particle.simplify_action_space = True

        if agent_name == "diayn":
            if domain == "igibson":
                cfg.agent.skill_dim = 64
            elif domain == "particle":
                cfg.agent.skill_dim = 8192
                cfg.snapshot_ts = 100000
            elif domain == "moma2d":
                cfg.agent.skill_dim = 125
            else:
                raise NotImplementedError

        return cfg

    def train(self):
        """
        Train a PPO agent with the given environment
        """

        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": self.cfg.total_timesteps,
            "env_name": self.cfg.domain,
        }
        project_name = self.cfg.domain + "_downstream_" + self.cfg.ds_task
        if self.cfg.factored:
            project_name += "_factored"

        run = wandb.init(
            project=project_name,
            config=config,
            group=self.cfg.experiment, name=self.cfg.experiment,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
        )

        if self.cfg.factored:
            from stable_baselines3 import FPPO # checkout https://github.com/JiahengHu/sb3-CausalMoMa
            assert self.cfg.domain == "particle"
            causal_matrix, reward_channels_dim = get_causal_matrix(self.cfg.domain, self.cfg.ds_task)
            kwargs = {}
            kwargs["clip_range"] = 0.2
            kwargs["target_kl"] = 0.15
            kwargs["normalize_advantage"] = True
            kwargs["gae_lambda"] = 0.95

            kwargs["sep_vnet"] = False
            kwargs["value_loss_normalization"] = False
            kwargs["value_grad_rescale"] = False
            kwargs["approx_var_gamma"] = False
            model = FPPO("MlpPolicy", self.train_env, reward_channels_dim, causal_matrix, verbose=1,
                         n_steps=self.cfg.n_steps, tensorboard_log=f"runs/{run.id}", device=self.device, **kwargs, )
        else:
            model = PPO("MlpPolicy", self.train_env, verbose=1, n_steps=self.cfg.n_steps,
                        tensorboard_log=f"runs/{run.id}", device=self.device)
        model.learn(
            total_timesteps=self.cfg.total_timesteps,
            callback=CustomCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            )
        )
        model.save("ppo_weight")
        run.finish()

    def test(self):
        """
        Load weight; test the high-level policy
        """

        if self.cfg.test_weight == "None":
            # Test stepping the environment
            init_obs = self.train_env.reset()
            print("init obs")
            print(init_obs)
            for i in range(20):
                a = self.train_env.action_space.sample()
                observation, reward, done, _ = self.train_env.step(a)
                self.train_env.render(mode="human")
                print(observation[-11:])
                print(reward)
                print(done)
        else:
            ppo_model_path = get_original_cwd() / Path(os.path.join('downstream', self.cfg.test_weight, 'ppo_higher.zip'))
            print(f"loading PPO weights from {ppo_model_path}")
            model = PPO.load(ppo_model_path)
            obs = self.eval_env.reset()
            total_reward = 0
            while True:
                action, _states = model.predict(obs, deterministic=True)

                obs, rewards, dones, info = self.eval_env.step(action)
                self.eval_env.render("human")
                print(f"action: {action}")
                print(f"rewards: {rewards}")
                print(f"obs: {obs}")
                total_reward += rewards
                if dones:
                    print(f"total_reward: {total_reward}")
                    total_reward = 0
                    obs = self.eval_env.reset()

    def load_agent(self, agent):
        """
        Load agent from snapshot
        Load actor, critic and diayn, return agent with the new weights
        """
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain = self.cfg.domain

        cfg = self.cfg
        low_path = self.cfg.low_path
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / low_path

        def try_load(seed):
            actor = snapshot_dir / str(
                seed) / f'actor_{self.cfg.snapshot_ts}.pt'
            actor = get_original_cwd() / actor

            print(f"loading snapshot {actor}")
            if not actor.exists():
                return None
            with actor.open('rb') as f:
                actor = torch.load(f, map_location=self.device)
            agent.actor.load_state_dict(actor)

            return agent

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None

@hydra.main(config_path='.', config_name='train')
def main(cfg):
    workspace = Workspace(cfg)
    if cfg.test:
        workspace.test()
    else:
        workspace.train()


if __name__ == '__main__':
    main()
