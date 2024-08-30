import torch as torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp


class CustomQNetwork(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomQNetwork, self).__init__(observation_space, action_space)

        # Extract network architecture and activation function from kwargs or set defaults
        net_arch = kwargs.get('net_arch', [32, 32])
        activation_fn = kwargs.get('activation_fn', nn.ReLU)

        # Define the Q-network (q_net)
        self.q_net = nn.Sequential(
            *create_mlp(observation_space.shape[0], action_space.n, net_arch, activation_fn),
            nn.Dropout(p=0.3)  # Add Dropout to prevent overfitting
        )

        # Clone q_net to create q_net_target
        self.q_net_target = nn.Sequential(*[layer for layer in self.q_net])  # Ensure identical architecture without dropout differences

        self.optimizer = optim.Adam(self.parameters(), lr=lr_schedule(1))

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return self.q_net(obs)

    def _predict(self, obs, deterministic=False):
        return torch.argmax(self.forward(obs), dim=1)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return self._predict(obs, deterministic), state



class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=32):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], features_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)  # Adding Dropout
        )

    def forward(self, observations):
        return self.net(observations)
