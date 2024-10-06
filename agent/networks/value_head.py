import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import utils
from functorch import combine_state_for_ensemble
from functorch import vmap


class FactoredValueHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=2, Q_range=None):
        super().__init__()

        self.n_skill_channels = output_size

        Q1_list = []
        Q2_list = []
        for i in range(self.n_skill_channels):
            sizes = [hidden_size] * (num_layers + 1) + [1]

            Q1_layers = []
            # Add a trunk to the Q network
            Q1_layers += [nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh()]
            for i in range(num_layers):
                Q1_layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            Q1_layers += [nn.Linear(sizes[-2], sizes[-1])]
            Q1_list.append(nn.Sequential(*Q1_layers))

            Q2_layers = []
            Q2_layers += [nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh()]
            for i in range(num_layers):
                Q2_layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            Q2_layers += [nn.Linear(sizes[-2], sizes[-1])]
            Q2_list.append(nn.Sequential(*Q2_layers))

        # Parameter Stacking for speedup
        fmodel_Q1, params_Q1, buffers_Q1 = combine_state_for_ensemble(Q1_list)
        self.Q1_params = [nn.Parameter(p) for p in params_Q1]
        self.Q1_buffers = [nn.Buffer(b) for b in buffers_Q1]

        for i, param in enumerate(self.Q1_params):
            self.register_parameter('Q1_param_' + str(i), param)

        for i, buffer in enumerate(self.Q1_buffers):
            self.register_buffer('Q1_buffer_' + str(i), buffer)

        self.Q1_model = vmap(fmodel_Q1)

        fmodel_Q2, params_Q2, buffers_Q2 = combine_state_for_ensemble(Q2_list)
        self.Q2_params = [nn.Parameter(p) for p in params_Q2]
        self.Q2_buffers = [nn.Buffer(b) for b in buffers_Q2]

        for i, param in enumerate(self.Q2_params):
            self.register_parameter('Q2_param_' + str(i), param)

        for i, buffer in enumerate(self.Q2_buffers):
            self.register_buffer('Q2_buffer_' + str(i), buffer)

        self.Q2_model = vmap(fmodel_Q2)

        self.Q_range = Q_range

    def forward(self, x):
        x = torch.swapaxes(x, 0, 1)  # (n_skill_channels, batch_size, embedding_size)
        Q1 = self.Q1_model(self.Q1_params, self.Q1_buffers, x)  # (n_skill_channels, batch_size, 1)
        Q2 = self.Q2_model(self.Q2_params, self.Q2_buffers, x)

        Q_out = torch.cat([Q1, Q2], dim=-1)
        Q_out = torch.swapaxes(Q_out, 0, 1)

        if self.Q_range is not None:
            Q_out = torch.tanh(Q_out) * self.Q_range
        return Q_out


class SeparateValueHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=2, Q_range=None):
        super().__init__()

        self.n_skill_channels = output_size

        Q1_list = []
        Q2_list = []
        for i in range(self.n_skill_channels):
            sizes = [hidden_size] * (num_layers + 1) + [1]

            Q1_layers = []
            # Add a trunk to the Q network
            Q1_layers += [nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh()]
            for i in range(num_layers):
                Q1_layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            Q1_layers += [nn.Linear(sizes[-2], sizes[-1])]
            Q1_list.append(nn.Sequential(*Q1_layers))

            Q2_layers = []
            Q2_layers += [nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh()]
            for i in range(num_layers):
                Q2_layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            Q2_layers += [nn.Linear(sizes[-2], sizes[-1])]
            Q2_list.append(nn.Sequential(*Q2_layers))

        # Parameter Stacking for speedup
        fmodel_Q1, params_Q1, buffers_Q1 = combine_state_for_ensemble(Q1_list)
        self.Q1_params = [nn.Parameter(p) for p in params_Q1]
        self.Q1_buffers = [nn.Buffer(b) for b in buffers_Q1]

        for i, param in enumerate(self.Q1_params):
            self.register_parameter('Q1_param_' + str(i), param)

        for i, buffer in enumerate(self.Q1_buffers):
            self.register_buffer('Q1_buffer_' + str(i), buffer)

        self.Q1_model = vmap(fmodel_Q1)

        fmodel_Q2, params_Q2, buffers_Q2 = combine_state_for_ensemble(Q2_list)
        self.Q2_params = [nn.Parameter(p) for p in params_Q2]
        self.Q2_buffers = [nn.Buffer(b) for b in buffers_Q2]

        for i, param in enumerate(self.Q2_params):
            self.register_parameter('Q2_param_' + str(i), param)

        for i, buffer in enumerate(self.Q2_buffers):
            self.register_buffer('Q2_buffer_' + str(i), buffer)

        self.Q2_model = vmap(fmodel_Q2)

        self.Q_range = Q_range

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.expand(self.n_skill_channels, -1, -1)

        Q1 = self.Q1_model(self.Q1_params, self.Q1_buffers, x)  # (n_skill_channels, batch_size, 1)
        Q2 = self.Q2_model(self.Q2_params, self.Q2_buffers, x)

        Q_out = torch.cat([Q1, Q2], dim=-1)
        Q_out = torch.swapaxes(Q_out, 0, 1)

        if self.Q_range is not None:
            Q_out = torch.tanh(Q_out) * self.Q_range

        return Q_out