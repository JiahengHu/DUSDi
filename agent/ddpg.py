from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.diayn_actors import Actor
from agent.diayn_critics import Critic
from agent.params import separate_sac_reward, Q_min_over_sum

import utils
from agent.partition_utils import obtain_partitions


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class DDPGAgent:
    def __init__(self,
                 name,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape,
                 lr,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 stddev_schedule,
                 nstep,
                 batch_size,
                 cuda_id,
                 stddev_clip,
                 init_critic,
                 use_tb,
                 use_wandb,
                 init_temperature,
                 update_alpha,
                 sac,
                 log_std_bounds,
                 critic_type,
                 meta_dim=0,
                 device=None,
                 domain=None):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device = torch.device("cuda:{}".format(cuda_id))
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        self.update_alpha = update_alpha
        self.sac = sac
        self.log_std_bounds = log_std_bounds
        self.domain = domain
        self.critic_type = critic_type

        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim

        self.actor = self.init_actor().to(device)

        self.attn_bal = False
        if critic_type == "mono":
            self.monolithic_Q = True
        else:
            if critic_type in ["ind", "attn", "sipat_max", "sipat_avg"]:
                self.partition_critic = True
            else:
                assert critic_type in ["sep", "mask_unwt", "mask_wt", "mask_topk", "branch"]
                if critic_type in ["mask_wt", "mask_topk"]:
                    self.attn_bal = True
                self.partition_critic = False
            self.monolithic_Q = False

        self.skill_obs_dependency, self.skill_action_dependency, self.skill_skill_dependency = None, None, None

        if self.monolithic_Q:
            self.log_multiplier = 1
        else:
            self.log_multiplier = self.skill_channel

        self.critic = self.make_critics(critic_type).to(self.device)
        self.critic_target = self.make_critics(critic_type).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        else:
            self.encoder_opt = None

        if self.sac:
            self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
            self.log_alpha.requires_grad = True
            # set target entropy to -|A|
            self.target_entropy = -action_shape[0]
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    # Different methods may have different actors, hence the need for this function
    def init_actor(self):
        actor = Actor(self.obs_type, self.obs_dim, self.action_dim,
                      self.feature_dim, self.hidden_dim, self.sac, self.log_std_bounds, self.domain)
        return actor

    def make_critics(self, critic_type):
        if critic_type == "mono":
            critic = Critic(self.obs_type, self.obs_dim, self.action_dim,
                                 self.feature_dim, self.hidden_dim)
        else:
            raise NotImplementedError
        return critic

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            raise NotImplementedError
            # First, we shouldn't init critic during downstream; second, this need to handle multiple Q
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        h = self.encoder(obs)
        inputs = [h]
        for key, value in meta.items():
            if key == "skill":
                value = torch.as_tensor(value, device=self.device)
                inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        self.dist = dist
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            if self.sac:
                next_action = dist.rsample()
                log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            else:
                next_action = dist.sample(clip=self.stddev_clip)

        # We update the Q function for each reward signal
        if self.monolithic_Q:
            with torch.no_grad():
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
                target_V = torch.min(target_Q1, target_Q2)
                if self.sac:
                    target_V -= self.alpha.detach() * log_prob
                assert reward.shape == target_V.shape
                target_Q = reward + (discount * target_V)

            Q1, Q2 = self.critic(obs, action)
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        else:
            if self.partition_critic:
                nxt_obs, nxt_skill = torch.split(
                    next_obs, [self.obs_dim - self.skill_dim * self.skill_channel,
                               self.skill_dim * self.skill_channel], dim=-1)
                nxt_obs_parts, nxt_skill_parts, nxt_action_parts = obtain_partitions(
                    nxt_obs, nxt_skill, next_action, self.domain, self.skill_dim, self.skill_channel)

                cur_obs, cur_skill = torch.split(
                    obs, [self.obs_dim - self.skill_dim * self.skill_channel,
                          self.skill_dim * self.skill_channel], dim=-1)
                # These dependencies will only be used for ind critic
                obs_parts, skill_parts, action_parts = obtain_partitions(
                    cur_obs, cur_skill, action, self.domain, self.skill_dim, self.skill_channel)


                with torch.no_grad():
                    target_out = self.critic_target(nxt_obs_parts, nxt_skill_parts, nxt_action_parts,
                                                    self.skill_obs_dependency, self.skill_action_dependency,
                                                    self.skill_skill_dependency)

                Q_out = self.critic(obs_parts, skill_parts, action_parts,
                                    self.skill_obs_dependency, self.skill_action_dependency, self.skill_skill_dependency)

            else:
                with torch.no_grad():
                    target_out = self.critic_target(next_obs, next_action)
                Q_out = self.critic(obs, action)

            # Processing target_out and Q_out is the same
            with torch.no_grad():
                if Q_min_over_sum:
                    target_out_sum = target_out.sum(dim=1)
                    _, min_idx = torch.min(target_out_sum, dim=1)
                    target_V = target_out[torch.arange(target_out.shape[0]), :, min_idx]

                else:
                    target_V, _ = torch.min(target_out, dim=-1)

                if self.sac:
                    if separate_sac_reward:
                        ent_r = -self.alpha.detach() * log_prob * discount
                        reward = torch.concat([reward, ent_r], dim=-1)
                    else:
                        target_V -= self.alpha.detach() * log_prob
                assert reward.shape == target_V.shape
                target_Q = reward + (discount * target_V)

            Q1 = Q_out[..., 0]
            Q2 = Q_out[..., 1]
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            if self.attn_bal:
                # Add the loss that regularize the attention score
                critic_loss += 0.2 * self.critic.bal_loss
            critic_loss *= self.skill_channel

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item() * self.log_multiplier
            metrics['critic_q1'] = Q1.mean().item() * self.log_multiplier
            metrics['critic_q2'] = Q2.mean().item() * self.log_multiplier
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if self.sac:
            action = dist.rsample()
        else:
            action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        if self.monolithic_Q:
            Q1, Q2 = self.critic(obs, action)
            Q = torch.min(Q1, Q2)
        else:
            if self.partition_critic:
                cur_obs, cur_skill = torch.split(
                    obs, [self.obs_dim - self.skill_dim * self.skill_channel,
                          self.skill_dim * self.skill_channel], dim=-1)
                obs_parts, skill_parts, action_parts = obtain_partitions(
                    cur_obs, cur_skill, action, self.domain, self.skill_dim, self.skill_channel)

                Q_out = self.critic(obs_parts, skill_parts, action_parts,
                                    self.skill_obs_dependency, self.skill_action_dependency, self.skill_skill_dependency)
            else:
                Q_out = self.critic(obs, action)

            if Q_min_over_sum:
                Q_sum = Q_out.sum(dim=1)
                Q = torch.min(Q_sum[:, 0], Q_sum[:, 1])
            else:
                Q, _ = torch.min(Q_out, dim=-1)
                Q *= self.skill_channel

        if self.sac:
            actor_loss = (self.alpha.detach() * log_prob - Q).mean()
        else:
            actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.sac and self.update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            metrics['alpha_loss'] = alpha_loss.item()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            if self.sac:
                metrics['alpha_value'] = self.alpha
                metrics['actor_ent'] = -log_prob.mean().item()
            else:
                metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
