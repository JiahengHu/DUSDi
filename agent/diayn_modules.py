import torch
import torch.nn as nn
import numpy as np
import utils
from agent.partition_utils import get_domain_stats, observation_filter
import torch.nn.utils.spectral_norm as spectral_norm

# Simplest multi-dim DIAYN
class MULTI_DIAYN(nn.Module):
    def __init__(self, domain, skill_dim, hidden_dim, skill_channel, env_config, use_spectral_norm):
        super().__init__()
        self.domain = domain
        self.env_config = env_config
        obs_dim, _ = get_domain_stats(domain, self.env_config)

        self.skill_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim,  skill_channel * skill_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        next_obs = observation_filter(next_obs, self.domain, self.env_config)
        skill_pred = self.skill_pred_net(next_obs)
        return skill_pred


class PARTED_DIAYN(nn.Module):
    def __init__(self, domain, skill_dim, hidden_dim, skill_channel, env_config, use_spectral_norm):
        super().__init__()

        self.domain = domain
        self.env_config = env_config
        obs_dim, state_partition_points = get_domain_stats(domain, self.env_config)
        assert len(state_partition_points) == skill_channel + 1

        self.obs_dim = obs_dim
        self.state_partition_points = state_partition_points
        self.skill_pred_nets = nn.ModuleList()
        for i in range(skill_channel):
            local_obs_size = state_partition_points[i + 1] - state_partition_points[i]

            if use_spectral_norm:
                skill_net = nn.Sequential(spectral_norm(nn.Linear(local_obs_size, hidden_dim)),
                                          nn.ReLU(),
                                          spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                                          nn.ReLU(),
                                          spectral_norm(nn.Linear(hidden_dim, skill_dim)))
            else:
                skill_net = nn.Sequential(nn.Linear(local_obs_size, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, skill_dim))
            self.skill_pred_nets.append(skill_net)

        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        # Partition the observations with different nets
        next_obs = observation_filter(next_obs, self.domain, self.env_config)
        assert next_obs.shape[1] == self.obs_dim

        output_list = []
        for i in range(len(self.state_partition_points) - 1):
            local_obs = next_obs[:, self.state_partition_points[i]:self.state_partition_points[i + 1]]
            skill_pred = self.skill_pred_nets[i](local_obs)
            output_list.append(skill_pred)
        skill_pred = torch.cat(output_list, dim=-1)
        return skill_pred


class PARTED_ANTI_DIAYN(nn.Module):
    """
    Anti predictor: this is used to predict z given all irrelevant state variables
    """
    def __init__(self, domain, skill_dim, hidden_dim, skill_channel, env_config):
        super().__init__()

        self.domain = domain
        self.env_config = env_config
        obs_dim, state_partition_points = get_domain_stats(domain, self.env_config)
        assert len(state_partition_points) == skill_channel + 1

        self.obs_dim = obs_dim
        self.skill_channel = skill_channel
        self.state_partition_points = state_partition_points
        self.skill_pred_nets = nn.ModuleList()
        for i in range(skill_channel):
            local_obs_size = state_partition_points[i + 1] - state_partition_points[i]
            local_anti_obs_size = obs_dim - local_obs_size
            assert local_anti_obs_size > 0
            skill_net = nn.Sequential(nn.Linear(local_anti_obs_size, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, skill_dim))
            self.skill_pred_nets.append(skill_net)

        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        next_obs = observation_filter(next_obs, self.domain, self.env_config)
        assert next_obs.shape[1] == self.obs_dim

        # Partition the observations with different nets
        output_list = []
        for i in range(self.skill_channel):
            local_obs_ind = np.concatenate([np.array(range(0, self.state_partition_points[i])),
                                            np.array(range(self.state_partition_points[i + 1], self.obs_dim))])
            local_obs = next_obs[:, local_obs_ind]
            skill_pred = self.skill_pred_nets[i](local_obs)
            output_list.append(skill_pred)
        skill_pred = torch.cat(output_list, dim=-1)
        return skill_pred


# Multi-dim DIAYN with transition (i.e. use two states instead of one)
class MULTI_TRANS_DIAYN(nn.Module):
    def __init__(self, domain, skill_dim, hidden_dim, skill_channel, env_config):
        super().__init__()

        self.domain = domain
        self.env_config = env_config
        obs_dim, _ = get_domain_stats(domain, self.env_config)

        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.state_net = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim//2))

        self.next_state_net = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim//2))

        self.pred_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, self.skill_dim * skill_channel))

        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        obs, next_obs = observation_filter(obs, self.domain, self.env_config), observation_filter(next_obs, self.domain, self.env_config)
        assert len(obs.size()) == len(next_obs.size())

        obs = self.state_net(obs)
        next_obs = self.state_net(next_obs)
        pred = self.pred_net(torch.cat([obs, next_obs], 1))
        return pred


class PARTED_TRANS_DIAYN(nn.Module):
    def __init__(self, domain, skill_dim, hidden_dim, skill_channel, env_config):
        super().__init__()
        self.domain = domain
        self.env_config = env_config
        obs_dim, state_partition_points = get_domain_stats(domain, self.env_config)
        assert len(state_partition_points) == skill_channel + 1

        self.obs_dim = obs_dim
        self.state_partition_points = state_partition_points
        self.skill_pred_nets = nn.ModuleList()

        for i in range(skill_channel):
            skill_net = nn.ModuleList()
            local_obs_size = state_partition_points[i + 1] - state_partition_points[i]

            state_net = nn.Sequential(nn.Linear(local_obs_size, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim // 2))

            next_state_net = nn.Sequential(nn.Linear(local_obs_size, hidden_dim), nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim // 2))

            pred_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, skill_dim))

            skill_net.append(state_net)
            skill_net.append(next_state_net)
            skill_net.append(pred_net)

            self.skill_pred_nets.append(skill_net)

        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        obs, next_obs = observation_filter(obs, self.domain, self.env_config), observation_filter(next_obs, self.domain, self.env_config)
        assert obs.shape[1] == self.obs_dim

        # Partition the observations with different nets
        output_list = []
        for i in range(len(self.state_partition_points) - 1):
            local_obs = obs[:, self.state_partition_points[i]:self.state_partition_points[i + 1]]
            local_next_obs = next_obs[:, self.state_partition_points[i]:self.state_partition_points[i + 1]]
            skill_net = self.skill_pred_nets[i]

            state = skill_net[0](local_obs)
            next_state = skill_net[1](local_next_obs)
            skill_pred = skill_net[2](torch.cat([state, next_state], 1))

            output_list.append(skill_pred)
        skill_pred = torch.cat(output_list, dim=-1)
        return skill_pred