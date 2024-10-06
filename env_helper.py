import os
import wrapper

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from custom_env.simple_dm_env import SimpleDMEnv
from custom_env.simple_gym_env import SimpleGymEnv
from custom_env.moma_2d_gym_env import MoMa2DGymEnv
import copy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Callable
from custom_env.moma_2d_downstream_env import MoMa2DGymDSEnv
from custom_env.hierarchical_env_wrapper import HierarchicalDiscreteEnv, HierarchicalContinuousEnv, HierarchicalDiaynEnv
from custom_env.hierarchical_env_wrapper import FlatEnvWrapper
import torch


try:
    from pettingzoo.mpe import simple_heterogenous_v3
    from pettingzoo.utils.wrappers.centralized_wrapper import (CentralizedWrapper,
                                                               DownstreamCentralizedWrapper,
                                                               SequentialDSWrapper)
except:
    print("no pettingzoo installation detected")


def load_img_encoder(device):
    from slot_attention.load_model import get_encoder
    # load image encoder which is used for particle env
    encoder = get_encoder(device)
    return encoder


def get_single_gym_env(cfg, rank=0):
    """
    :param cfg:
    :param rank:
    :return: a gym environment
    """

    if cfg.domain == "toy":
        env = SimpleGymEnv(max_step=cfg.env.toy.episode_length, stochastic=cfg.env.toy.stochastic,
                           limit=cfg.env.toy.limit)
    elif cfg.domain == "moma2d":
        env = MoMa2DGymEnv(max_step=cfg.env.moma2d.episode_length, show_empty=cfg.env.moma2d.show_empty)
        env.seed(cfg.seed + rank)
    elif cfg.domain == "particle":
        N = cfg.env.particle.N

        # Only in test_skills
        if "render" in cfg and cfg.render:
            render_mode = "human"
        else:
            render_mode = "rgb_array"

        if cfg.env.particle.use_img:
            img_encoder = load_img_encoder(torch.device("cuda:{}".format(cfg.cuda_id)))
        else:
            img_encoder = None

        env = simple_heterogenous_v3.parallel_env(
            render_mode=render_mode,
            max_cycles=1000,
            continuous_actions=True,
            local_ratio=0,
            N=N,
            img_encoder=img_encoder
        )
        env = CentralizedWrapper(env, simplify_action_space=cfg.env.particle.simplify_action_space)
        env.reset(seed=cfg.seed + rank)
    elif cfg.domain in ["igibson", "wipe"]:
        if cfg.domain == "wipe":
            physics_timestep = 1 / 120.0
            config_fn = "tiago_wipe.yaml"
        elif cfg.domain == "igibson":
            physics_timestep = 1 / 60.0
            config_fn = "fetch_skill.yaml"
        config_filename = os.path.join(igibson.configs_path, config_fn)
        config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        if "update_skill_every_step" in cfg.agent:
            config_data["switch_skill_frequency"] = cfg.agent.update_skill_every_step
        else:
            # multi-diayn config
            if cfg.agent.name in ["dusdi_diayn"]:
                config_data["switch_skill_frequency"] = cfg.agent.training_params[cfg.domain].update_skill_every_step
            else:
                config_data["switch_skill_frequency"] = 200

        config_data["return_dist"] = False

        if "vis" in cfg:
            config_data["load_texture"] = True
            mode = cfg.vis

            # add object if wiping environment
            if cfg.domain == "wipe":
                config_data.pop('load_object_categories', None)
                config_data.pop('load_room_types', None)
                config_data["board_cover"] = True

        else:
            mode = "headless"

        env = iGibsonEnv(
            config_file=config_data,
            mode=mode,
            action_timestep=1 / 10.0,
            physics_timestep=physics_timestep,
        )
        env.seed(cfg.seed + rank)
    else:
        raise NotImplementedError

    set_random_seed(cfg.seed + rank)
    return env


def make_envs(cfg):
    # Examples for creating DM environment
    if cfg.domain == "toy" and not cfg.env.toy.gym:
        use_gym = cfg.env.toy.gym
        assert cfg.n_env == 1
        train_env = wrapper.make_simple(
            SimpleDMEnv(max_step=cfg.env.toy.episode_length, stochastic=cfg.stochastic, limit=cfg.limit))
        eval_env = wrapper.make_simple(
            SimpleDMEnv(max_step=cfg.env.toy.episode_length, stochastic=cfg.stochastic, limit=cfg.limit))

    else:
        use_gym = True
        if cfg.n_env == 1:
            train_env = get_single_gym_env(cfg)
        else:
            # Multiprocess
            env_fns = [lambda: get_single_gym_env(cfg, rank) for rank in range(cfg.n_env)]
            train_env = SubprocVecEnv(env_fns)

        # Initialize the evaluation environment
        eval_env = get_single_gym_env(cfg)

        train_env = wrapper.make_simple_gym(train_env, n_env=cfg.n_env)
        eval_env = wrapper.make_simple_gym(eval_env)

    return train_env, eval_env, use_gym


# Used for evaluation
def make_eval_envs(cfg):
    eval_env = get_single_gym_env(cfg)
    eval_env = wrapper.make_simple_gym(eval_env)
    return eval_env


# Decide how to wrap the DS environment based on the type of algorithm
def wrap_ds_env(env, cfg, actor, device, low_level_step, vis=False):
    if cfg.agent.name == "dusdi_diayn":
        env = HierarchicalDiscreteEnv(env, cfg.agent.skill_channel, cfg.agent.skill_dim, low_level_step,
                                      device, actor, vis)
    elif cfg.agent.name == "diayn":
        env = HierarchicalDiaynEnv(env, cfg.agent.skill_dim, low_level_step,
                                   device, actor, vis)
    elif cfg.agent.name == "cic":
        env = HierarchicalContinuousEnv(env, 64, low_level_step,
                                        device, actor, vis)
    elif cfg.agent.name in ["rnd", "icm", "ddpg"]:
        env = FlatEnvWrapper(env, low_level_step)
        env = wrapper.make_simple_gym(env)
    else:
        raise NotImplementedError
    return env


def make_ds_envs(cfg, actor, device):
    max_step = 1000
    if cfg.domain == "moma2d":
        low_level_step = 50
        def make_moma2d_envs(vis=False):
            env = MoMa2DGymDSEnv(max_step, show_empty=cfg.env.moma2d.show_empty, version=cfg.ds_task)
            env = wrap_ds_env(env, cfg, actor, device, low_level_step, vis)
            return env
        return make_moma2d_envs

    elif cfg.domain == "particle":
        low_level_step = 50
        def make_particle_ds_env(vis=False):
            N = cfg.env.particle.N
            env = simple_heterogenous_v3.parallel_env(
                render_mode='rgb_array',
                max_cycles=max_step,
                continuous_actions=True,
                local_ratio=0,
                N=N
            )

            if cfg.ds_task == "poison_s":
                env = DownstreamCentralizedWrapper(env, landmark_id=[1, 2], N=N, factorize=cfg.factored,
                                                   simplify_action_space=cfg.env.particle.simplify_action_space)
            elif cfg.ds_task == "poison_m":
                env = DownstreamCentralizedWrapper(env, landmark_id=[0, 2, 4, 6, 9], N=N, factorize=cfg.factored,
                                                   simplify_action_space=cfg.env.particle.simplify_action_space)
            elif cfg.ds_task == "poison_l":
                env = DownstreamCentralizedWrapper(env, landmark_id=[0, 1, 2, 3, 4, 6, 7, 9], N=N,
                                                   factorize=cfg.factored,
                                                   simplify_action_space=cfg.env.particle.simplify_action_space)

            elif cfg.ds_task == "sequential":
                env = SequentialDSWrapper(env, N=N, agent_sequence=[0, 2, 4, 6, 9],
                                          simplify_action_space=cfg.env.particle.simplify_action_space)
            else:
                raise NotImplementedError

            env = wrap_ds_env(env, cfg, actor, device, low_level_step, vis)
            return env

        return make_particle_ds_env

    elif cfg.domain == "igibson":
        low_level_step = 100

        # Function callback to create environments
        def make_igibson_downstream_env(vis=False):
            env = get_single_gym_env(cfg)

            if cfg.ds_task == "eye":
                env = DownstreamEyeWrapper(env)
            elif cfg.ds_task == "eyebase":
                env = DownstreamBaseEyeWrapper(env)
            elif cfg.ds_task == "full":
                env = DownstreamFullWrapper(env)
            else:
                raise NotImplementedError

            env = wrap_ds_env(env, cfg, actor, device, low_level_step, vis)
            return env

        return make_igibson_downstream_env

    else:
        assert NotImplementedError


if __name__ == "__main__":
    from pettingzoo.mpe import simple_heterogenous_v3
    from pettingzoo.utils.wrappers.centralized_wrapper import CentralizedWrapper, DownstreamCentralizedWrapper
    from custom_env.hierarchical_env_wrapper import FlatEnvWrapper
    N = 10
    landmark_id = range(N)
    low_level_step = 50
    env = simple_heterogenous_v3.parallel_env(
        render_mode='rgb_array',
        max_cycles=1000,
        continuous_actions=True,
        local_ratio=0,
        N=N
    )
    env = DownstreamCentralizedWrapper(env, landmark_id=landmark_id, N=N,
                                       simplify_action_space=True)
    env = FlatEnvWrapper(env, low_level_step)
    init_obs = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

