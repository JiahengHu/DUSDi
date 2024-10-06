import torch
import torch.nn as nn

from .modules.rgb_modules import *
from .modules.language_modules import *
from .modules.transformer_modules import *
from .base_policy import BasePolicy
from .value_head import *
from ..partition_utils import get_env_factorization

###############################################################################
#
# Q should take in action, skill & observation, and output Q values, where Q = n skill channels
# 		- We don't need dummy token in this case, we can just query the Q value for each skill channel
#
###############################################################################


class StatePartitionTokens(nn.Module):
    def __init__(
        self,
        skill_dim,
        skill_channel,
        domain="particle",
        obs_partition=None,
        skill_partition=None,
        extra_num_layers=0,
        extra_hidden_size=64,
        extra_embedding_size=32,
    ):
        """
        Maps state factors and skill factors into tokens
        """
        super().__init__()
        self.extra_embedding_size = extra_embedding_size

        self.obs_partition, self.skill_partition, self.action_partition = get_env_factorization(
            domain, skill_dim, skill_channel)
        self.skill_dim = skill_dim
        self.skill_channel = skill_channel

        # We can also try more coarse partitions - maybe later
        self.num_extra = len(self.obs_partition) + len(self.skill_partition) + len(self.action_partition)

        self.extra_encoders_list = []

        def generate_proprio_mlp_fn(factor_size):
            assert factor_size > 0  # we indeed have extra information
            if extra_num_layers > 0:
                layers = [nn.Linear(factor_size, extra_hidden_size)]
                for i in range(1, extra_num_layers):
                    layers += [
                        nn.Linear(extra_hidden_size, extra_hidden_size),
                        nn.ReLU(inplace=True),
                    ]
                layers += [nn.Linear(extra_hidden_size, extra_embedding_size)]
            else:
                layers = [nn.Linear(factor_size, extra_embedding_size)]

            proprio_mlp = nn.Sequential(*layers)
            self.extra_encoders_list.append(proprio_mlp)

        for dim in self.skill_partition + self.obs_partition + self.action_partition:
            generate_proprio_mlp_fn(dim)

        self.encoders = nn.ModuleList(self.extra_encoders_list)

    def forward(self, obs_list):
        """
        obs_list: a list of [B, k], where k is specified by the obs_partition
        map above to a latent vector of shape (B, num_factors, H)
        """
        tensor_list = []

        for idx in range(len(obs_list)):
            tensor_list.append(
                self.encoders[idx](
                    obs_list[idx]
                )
            )

        x = torch.stack(tensor_list, dim=-2)
        return x


class PerturbationAttention:
    """
    See https://arxiv.org/pdf/1711.00138.pdf for perturbation-based visualization
    for understanding a control agent.
    """

    def __init__(self, model, image_size=[128, 128], patch_size=[16, 16], device="cpu"):

        self.model = model
        self.patch_size = patch_size
        H, W = image_size
        num_patches = (H * W) // np.prod(patch_size)
        # pre-compute mask
        h, w = patch_size
        nh, nw = H // h, W // w
        mask = (
            torch.eye(num_patches)
            .view(num_patches, num_patches, 1, 1)
            .repeat(1, 1, patch_size[0], patch_size[1])
        )  # (np, np, h, w)
        mask = rearrange(
            mask.view(num_patches, nh, nw, h, w), "a b c d e -> a (b d) (c e)"
        )  # (np, H, W)
        self.mask = mask.to(device).view(1, num_patches, 1, H, W)
        self.num_patches = num_patches
        self.H, self.W = H, W
        self.nh, self.nw = nh, nw

    def __call__(self, data):
        rgb = data["obs"]["agentview_rgb"]  # (B, C, H, W)
        B, C, H, W = rgb.shape

        rgb_ = rgb.unsqueeze(1).repeat(1, self.num_patches, 1, 1, 1)  # (B, np, C, H, W)
        rgb_mean = rgb.mean([2, 3], keepdims=True).unsqueeze(1)  # (B, 1, C, 1, 1)
        rgb_new = (rgb_mean * self.mask) + (1 - self.mask) * rgb_  # (B, np, C, H, W)
        rgb_stack = torch.cat([rgb.unsqueeze(1), rgb_new], 1)  # (B, 1+np, C, H, W)

        rgb_stack = rearrange(rgb_stack, "b n c h w -> (b n) c h w")
        res = self.model(rgb_stack).view(B, self.num_patches + 1, -1)  # (B, 1+np, E)
        base = res[:, 0].view(B, 1, -1)
        others = res[:, 1:].view(B, self.num_patches, -1)

        attn = F.softmax(1e5 * (others - base).pow(2).sum(-1), -1)  # (B, num_patches)
        attn_ = attn.view(B, 1, self.nh, self.nw)
        attn_ = (
            F.interpolate(attn_, size=(self.H, self.W), mode="bilinear")
            .detach()
            .cpu()
            .numpy()
        )
        return attn_


###############################################################################
#
# A Transformer Policy
#
###############################################################################

class AttnValue(BasePolicy):
    """
    Input: k * action tokens; n * state factors; m * skill factors
    Output: a_t or distribution of a_t
    """

    def __init__(self, cfg, skill_dim, skill_channel):
        super().__init__(cfg)

        # All of these should be named value
        value_cfg = cfg
        embed_size = value_cfg.embed_size
        self.position_encoding = cfg.use_position_encoding

        self.extra_encoder = StatePartitionTokens(
            skill_dim=skill_dim,
            skill_channel=skill_channel,
            extra_num_layers=value_cfg.extra_num_layers,
            extra_hidden_size=value_cfg.extra_hidden_size,
            extra_embedding_size=embed_size,
        )

        value_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
        self.temporal_position_encoding_fn = eval(
            value_cfg.temporal_position_encoding.network
        )(**value_cfg.temporal_position_encoding.network_kwargs)

        self.transformer_decoder = TransformerDecoder(
            input_size=embed_size,
            num_layers=value_cfg.transformer_num_layers,
            num_heads=value_cfg.transformer_num_heads,
            head_output_size=value_cfg.transformer_head_output_size,
            mlp_hidden_size=value_cfg.transformer_mlp_hidden_size,
            dropout=value_cfg.transformer_dropout,
        )

        value_head_kwargs = value_cfg.value_head.network_kwargs
        value_head_kwargs.input_size = embed_size
        value_head_kwargs.output_size = self.extra_encoder.skill_channel

        self.value_head = eval(value_cfg.value_head.network)(
            **value_cfg.value_head.network_kwargs
        )

        self.latent_queue = []
        self.max_seq_len = value_cfg.transformer_max_seq_len

        # Get env params from the encoder
        self.action_partition = self.extra_encoder.action_partition
        self.num_action_factor = len(self.action_partition)
        self.skill_dim = self.extra_encoder.skill_dim
        self.skill_channel = self.extra_encoder.skill_channel
        self.skill_partition = self.extra_encoder.skill_partition
        self.obs_partition = self.extra_encoder.obs_partition

        # One token per action dimension
        action_token = nn.Parameter(torch.randn([self.num_action_factor, embed_size]))
        self.register_parameter("action_token", action_token)

    def transformer_forward(self, x):
        if self.position_encoding:
            pos_emb = self.temporal_position_encoding_fn(x)  # (num_modalities, E)
            x = x + pos_emb  # (B, num_modality, E)

        # TODO: we can consider adding masks later
        # self.temporal_transformer.compute_mask(x.shape)

        x = self.transformer_decoder(x)

        # This is getting the first K token, which are the skill token
        return x[:, :self.skill_channel]  # (B, N, E)

    def spatial_encode(self, obs_parts, skill_parts, action_parts):
        encoded = self.extra_encoder(skill_parts + obs_parts + action_parts)  # (B, num_modalities, E)
        return encoded

    def forward(self, obs_parts, skill_parts, action_parts, skill_obs_dependency, skill_action_dependency):
        x = self.spatial_encode(obs_parts, skill_parts, action_parts)
        x = self.transformer_forward(x)
        values = self.value_head(x)
        return values

    def reset(self):
        self.latent_queue = []
