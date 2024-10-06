# import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

REGISTERED_POLICIES = {}


def register_policy(policy_class):
    """Register a policy class with the registry."""
    policy_name = policy_class.__name__.lower()
    if policy_name in REGISTERED_POLICIES:
        raise ValueError("Cannot register duplicate policy ({})".format(policy_name))

    REGISTERED_POLICIES[policy_name] = policy_class


def get_policy_class(policy_name):
    """Get the policy class from the registry."""
    if policy_name.lower() not in REGISTERED_POLICIES:
        raise ValueError(
            "Policy class with name {} not found in registry".format(policy_name)
        )
    return REGISTERED_POLICIES[policy_name.lower()]


def get_policy_list():
    return REGISTERED_POLICIES


class PolicyMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all policies that should not be registered here.
        _unregistered_policies = ["BasePolicy"]

        if cls.__name__ not in _unregistered_policies:
            register_policy(cls)
        return cls


class BasePolicy(nn.Module, metaclass=PolicyMeta):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, data):
        """
        The forward function for training.
        """
        raise NotImplementedError

    def get_action(self, data):
        """
        The api to get policy's action.
        """
        raise NotImplementedError

    def _get_img_tuple(self, data):
        img_tuple = tuple(
            [data["obs"][img_name] for img_name in self.image_encoders.keys()]
        )
        return img_tuple

    def _get_aug_output_dict(self, out):
        img_dict = {
            img_name: out[idx]
            for idx, img_name in enumerate(self.image_encoders.keys())
        }
        return img_dict

    def compute_loss(self, data, reduction="mean"):
        data = self.preprocess_input(data, train_mode=True)
        dist = self.forward(data)
        loss = self.policy_head.loss_fn(dist, data["actions"], reduction)
        return loss

    def reset(self):
        """
        Clear all "history" of the policy if there exists any.
        """
        pass
