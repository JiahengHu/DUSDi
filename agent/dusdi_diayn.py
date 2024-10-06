import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent
from agent.diayn_modules import MULTI_DIAYN, MULTI_TRANS_DIAYN, PARTED_DIAYN, PARTED_TRANS_DIAYN, PARTED_ANTI_DIAYN
from agent.diayn_actors import SkillActor, MCPActor, SeparateSkillActor
from agent.diayn_critics import Critic, SepCritic, StateMaskCritic
from agent.networks.attention_value import AttnValue
from agent.networks.attention_policy import AttnPolicy
from agent.partition_utils import get_domain_stats


class DUSDI_Agent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale, parted, trans, use_spectral_norm,
                 update_skill_inter_episode, update_encoder, pred_hidden_dim, actor_type, anti, use_gate,
                 update_skill_threshold, anti_coef, training_params, add_task_reward, Q_range,
                 ind_type, env_config, mask_out_transitions, step_count_threshold, policy_config, value_config, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.update_skill_inter_episode = update_skill_inter_episode
        self.diayn_scale = diayn_scale
        self.update_encoder = update_encoder
        self.env_config = env_config
        self.actor_type = actor_type
        self.parted = parted
        self.anti = anti
        self.anti_coef_max = anti_coef
        self.anti_coef = 0
        self.use_gate = use_gate
        self.ind_type = ind_type
        self.policy_cfg = policy_config
        self.value_cfg = value_config
        self.add_task_reward = add_task_reward
        self.Q_range = Q_range

        self.update_skill_threshold = update_skill_threshold * 1000000

        self.mask_out_transitions = mask_out_transitions
        self.step_count_threshold = step_count_threshold
        self.start_updating_skill = False
        self.domain = kwargs["domain"]

        self.init_params()
        # increase obs shape to include skill dim
        # Essentially an n-hot vector, with n = channels
        self.meta_dim = kwargs["meta_dim"] = self.skill_dim * self.skill_channel

        # create actor and critic
        super().__init__(**kwargs)

        if anti:
            assert parted and not trans     # only support these for now

        # We can just pass in diayn args together
        diayn_args = [self.domain, self.skill_dim, pred_hidden_dim, self.diayn_skill_channel, env_config]

        # create diayn
        if self.diayn_skill_channel == 0:
            self.diayn = None
        else:
            if trans:
                if parted:
                    self.diayn = PARTED_TRANS_DIAYN(*diayn_args).to(self.device)
                else:
                    self.diayn = MULTI_TRANS_DIAYN(*diayn_args).to(self.device)
            else:
                if parted:
                    self.diayn = PARTED_DIAYN(*diayn_args, use_spectral_norm=use_spectral_norm).to(self.device)
                    if anti:
                        self.anti_diayn = PARTED_ANTI_DIAYN(*diayn_args).to(self.device)
                else:
                    self.diayn = MULTI_DIAYN(*diayn_args).to(self.device)

            # optimizers
            self.diayn_opt = torch.optim.Adam(self.diayn.parameters(), lr=self.lr)
            self.diayn.train()

        # loss criterion
        self.diayn_criterion = nn.CrossEntropyLoss()

        if anti:
            self.anti_diayn_opt = torch.optim.Adam(self.anti_diayn.parameters(), lr=self.lr)
            self.anti_diayn.train()

    def init_params(self):
        _, partitions = get_domain_stats(self.domain, self.env_config)
        self.diayn_skill_channel = len(partitions) - 1
        self.skill_channel = self.diayn_skill_channel

    # Actor of diayn is different from that of DDPG
    def init_actor(self):
        # Some policies only support partition
        partition_policy = ["mcp", "sep", "ind", "attn"]
        if self.actor_type in partition_policy:
            assert self.parted

        if self.actor_type == "skill":
            actor = SkillActor(self.obs_type, self.obs_dim, self.action_dim,
                      self.feature_dim, self.hidden_dim, self.sac, self.log_std_bounds, self.meta_dim)
        elif self.actor_type == "mcp":
            # MCP is not possible without a partition
            actor = MCPActor(self.obs_type, self.obs_dim, self.action_dim, self.feature_dim, self.hidden_dim,
                             self.sac, self.log_std_bounds, self.skill_channel, self.skill_dim, self.use_gate)
        elif self.actor_type == "sep":
            actor = SeparateSkillActor(self.obs_type, self.obs_dim, self.action_dim, self.feature_dim, self.hidden_dim,
                                       self.sac, self.log_std_bounds, self.skill_channel, self.skill_dim)
        elif self.actor_type == "attn":
            actor = AttnPolicy(self.policy_cfg, self.sac, skill_dim=self.skill_dim, skill_channel=self.skill_channel)
        else:
            assert self.actor_type == "mono"
            actor = super().init_actor()
        return actor

    # only multi-diayn supports all these different critics
    def make_critics(self, critic_type):
        if self.add_task_reward:
            # The other critics doesn't implement this
            assert critic_type in ["mask_unwt", "sep", "branch"]
            ext_r_dim = 1
        else:
            ext_r_dim = 0
        if critic_type == "mono":
            critic = Critic(self.obs_type, self.obs_dim, self.action_dim,
                                 self.feature_dim, self.hidden_dim)
        elif critic_type == "sep":
            critic = SepCritic(self.obs_type, self.obs_dim, self.action_dim,
                                 self.feature_dim, self.hidden_dim, self.skill_channel,
                               use_sac=self.sac, ext_r_dim=ext_r_dim)
        elif critic_type == "mask_unwt":
            critic = StateMaskCritic(self.obs_dim, self.action_dim, self.skill_dim, self.skill_channel, self.domain,
                                     self.device, weighted=False, use_sac=self.sac, Q_range=self.Q_range,
                                     topk_gating=False, attn_balancing=False, ext_r_dim=ext_r_dim)
        elif critic_type == "mask_wt":
            critic = StateMaskCritic(self.obs_dim, self.action_dim, self.skill_dim, self.skill_channel, self.domain,
                                     self.device, weighted=True, use_sac=self.sac, Q_range=self.Q_range,
                                     topk_gating=False, attn_balancing=True, ext_r_dim=ext_r_dim)
        else:
            raise NotImplementedError
        return critic

    def get_meta_specs(self):
        return (specs.Array((self.skill_channel, self.skill_dim,), np.float32, 'skill'),
                specs.Array((1,), np.int32, 'step'))

    def init_meta(self, num_envs=1):
        skill = np.zeros((num_envs, self.skill_channel, self.skill_dim), dtype=np.float32)
        for i in range(num_envs):
            skill[i][range(self.skill_channel), np.random.choice(self.skill_dim, size=self.skill_channel)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill.reshape(num_envs, self.skill_dim * self.skill_channel)
        meta['step'] = np.zeros([num_envs])
        return meta

    def get_meta_from_skill(self, skill_idx, num_envs):
        skill = np.zeros((self.skill_channel, self.skill_dim), dtype=np.float32)
        skill[range(self.skill_channel), skill_idx] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill.reshape(num_envs, self.skill_dim * self.skill_channel)
        return meta

    def get_random_skill(self):
        return np.random.choice(self.skill_dim, size=self.skill_channel)

    def update_meta(self, meta, step_count, time_step, n_env, total_step):
        update_skill_threshold = self.update_skill_threshold
        if self.update_skill_inter_episode and total_step >= update_skill_threshold:
            self.start_updating_skill = True
        else:
            self.start_updating_skill = False

        if self.start_updating_skill and step_count % self.update_skill_every_step == 0:
            new_meta = self.init_meta(n_env)
            # small fix; shouldn't matter
            new_meta['step'] = np.ones([n_env]) * step_count
        else:
            new_meta = meta.copy()
            new_meta['step'] = np.ones([n_env]) * step_count
        return new_meta

    def update_diayn(self, skill, obs, next_obs):
        metrics = dict()

        if self.diayn_skill_channel == 0:
            return metrics

        loss, df_accuracy, df_anti_accuracy = self.compute_diayn_loss(obs, next_obs, skill)

        self.diayn_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        if self.anti:
            self.anti_diayn_opt.zero_grad()
        loss.backward()
        self.diayn_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        if self.anti:
            self.anti_diayn_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['diayn_loss'] = loss.item()
            for idx, acc in enumerate(df_accuracy):
                metrics['diayn_acc_{}'.format(idx)] = acc.item()
            if self.anti:
                metrics['anti_diayn_acc'] = df_anti_accuracy
            metrics['diayn_los_avg'] = df_accuracy.mean().item()
        
        return metrics

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.diayn, self.diayn)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
        if self.anti:
            utils.hard_update_params(other.anti_diayn, self.anti_diayn)

    def compute_intr_reward(self, skill, obs, next_obs):

        # Skip if only gc
        if self.diayn_skill_channel == 0:
            return 0.0

        skill = skill.reshape(-1, self.skill_dim)  # (bs * channel) * dim
        z_hat = torch.argmax(skill, dim=-1)
        d_pred = self.diayn(obs, next_obs).reshape(-1, self.skill_dim)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=-1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=-1, keepdim=True)

        if not self.anti:
            reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                        z_hat] - math.log(1 / self.skill_dim)
        else:
            d_anti_pred = self.anti_diayn(obs, next_obs).reshape(-1, self.skill_dim)
            d_anti_pred_log_softmax = F.log_softmax(d_anti_pred, dim=-1)
            _, anti_pred_z = torch.max(d_anti_pred_log_softmax, dim=-1, keepdim=True)

            reward = (d_pred_log_softmax[torch.arange(d_pred.shape[0]), z_hat] - math.log(1 / self.skill_dim)
                      - (d_anti_pred_log_softmax[torch.arange(d_anti_pred.shape[0]),
                      z_hat] - math.log(1 / self.skill_dim)) * self.anti_coef)  # (bs * channel)

        if self.monolithic_Q:
            # reward is the mean over skill channels
            reward = reward.reshape(-1, self.diayn_skill_channel).sum(dim=-1, keepdims=True)
        else:
            reward = reward.reshape(-1, self.diayn_skill_channel)

        return reward * self.diayn_scale

    def compute_diayn_loss(self, state, next_state, skill):
        """
        DF Loss
        """
        # We merge the skill channel and the batch dim when computing loss

        if self.diayn_skill_channel == 0:
            return 0.0, 0.0, 0.0

        skill = skill.reshape(-1,  self.skill_dim)
        z_hat = torch.argmax(skill, dim=-1)
        d_pred = self.diayn(state, next_state).reshape(-1, self.skill_dim)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=-1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=-1, keepdim=True)

        d_loss = self.diayn_criterion(d_pred, z_hat) * self.diayn_skill_channel  # to maintain the original lr

        acc_list = torch.eq(
            z_hat, pred_z.reshape(1, list(pred_z.size())[0])[0]
        ).reshape(-1, self.diayn_skill_channel)
        df_accuracy = torch.sum(acc_list, dim=0).float() / acc_list.shape[0]

        df_anti_accuracy = None

        if self.anti:
            d_anti_pred = self.anti_diayn(state, next_state).reshape(-1, self.skill_dim)
            d_anti_pred_log_softmax = F.log_softmax(d_anti_pred, dim=-1)
            _, anti_pred_z = torch.max(d_anti_pred_log_softmax, dim=-1, keepdim=True)

            d_anti_loss = self.diayn_criterion(d_anti_pred, z_hat)

            d_loss = d_loss + d_anti_loss

            df_anti_accuracy = torch.sum(
                torch.eq(
                    z_hat, anti_pred_z.reshape(1, list(anti_pred_z.size())[0])[0]
                )
            ).float() / list(anti_pred_z.size())[0]

        return d_loss, df_accuracy, df_anti_accuracy

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        # update anti coeff
        anti_threshold = 1000000
        if self.anti and step >= anti_threshold:
            self.anti_coef = self.anti_coef_max

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill, step_count = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        # Masking out states during training
        if self.mask_out_transitions:  #  and self.update_skill_inter_episode
            residual_step_count = step_count % self.update_skill_every_step
            mask = residual_step_count > self.step_count_threshold
        else:
            mask = torch.ones_like(step_count, dtype=torch.bool)

        if self.reward_free:
            metrics.update(self.update_diayn(skill[mask], obs[mask], next_obs[mask]))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, obs, next_obs)
                intr_reward[~mask] = 0.0

            if self.use_tb or self.use_wandb:
                metrics['diayn_intr_reward'] = intr_reward.mean().item() * self.log_multiplier
            reward = intr_reward

            if self.add_task_reward:
                reward = torch.cat([reward, extr_reward], dim=-1)
        else:
            reward = extr_reward
            raise NotImplementedError

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()
            metrics["anti_coef"] = self.anti_coef

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
