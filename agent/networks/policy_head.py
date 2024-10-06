# import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import utils


class DeterministicHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=2):

        super().__init__()
        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if self.action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y


class FactoredHead(nn.Module):
    def __init__(self, input_size, output_size, sac, log_std_bounds, loss_coef, hidden_size=1024, num_layers=2):
        super().__init__()

        self.sac = sac
        if not self.sac:
            raise NotImplementedError

        self.act_par_num = len(output_size)

        policy_list = []
        for i in range(self.act_par_num):
            if sac:
                action_length = output_size[i] * 2
            else:
                action_length = output_size[i]

            sizes = [input_size] + [hidden_size] * num_layers + [action_length]
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]

            policy_list.append(nn.Sequential(*layers))

        self.primitives = nn.ModuleList(policy_list)
        self.log_std_bounds = log_std_bounds

    def forward(self, x, std=None):

        if self.sac:
            outs = []
            for i in range(self.act_par_num):
                action = self.primitives[i](x[:, i])
                act, sig = torch.chunk(action, 2, dim=-1)
                outs.append([act, sig])

            mus, var = zip(*outs)
            mu = torch.concat(mus, dim=-1)
            log_std = torch.concat(var, dim=-1)

            # constrain log_std inside [log_std_min, log_std_max]
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std + 1) * (log_std_max - log_std_min)
            std = log_std.exp()
            dist = utils.SquashedNormal(mu, std)
        else:
            outs = [self.primitives[i](x[:, i]) for i in range(self.act_par_num)]
            mu = torch.concat(outs, dim=-1)
            mu = torch.tanh(mu)
            dist = utils.TruncatedNormal(mu, std)
        return dist

    def loss_fn(self, gmm, target, reduction="mean"):
        raise NotImplementedError
        log_probs = gmm.log_prob(target)
        loss = -log_probs
        if reduction == "mean":
            return loss.mean() * self.loss_coef
        elif reduction == "none":
            return loss * self.loss_coef
        elif reduction == "sum":
            return loss.sum() * self.loss_coef
        else:
            raise NotImplementedError


class GMMHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        min_std=0.0001,
        num_modes=5,
        activation="softplus",
        low_eval_noise=False,
        # loss_kwargs
        loss_coef=1.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp

    def forward_fn(self, x):
        # x: (B, input_size)
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_size)
        means = torch.tanh(means)
        logits = self.logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_size
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits

    def forward(self, x):
        if x.ndim == 3:
            means, scales, logits = TensorUtils.time_distributed(x, self.forward_fn)
        elif x.ndim < 3:
            means, scales, logits = self.forward_fn(x)

        compo = D.Normal(loc=means, scale=scales)
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)
        gmm = D.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        return gmm

    def loss_fn(self, gmm, target, reduction="mean"):
        log_probs = gmm.log_prob(target)
        loss = -log_probs
        if reduction == "mean":
            return loss.mean() * self.loss_coef
        elif reduction == "none":
            return loss * self.loss_coef
        elif reduction == "sum":
            return loss.sum() * self.loss_coef
        else:
            raise NotImplementedError
