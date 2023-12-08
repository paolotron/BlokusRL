import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BaseModel
import gym
import torch
from stable_baselines3.common.type_aliases import PyTorchObs


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        return y + x


class BlokusSeer(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, num_cnn_blocks=3, hidden_dim=128):
        super().__init__(observation_space, features_dim=4)

        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 3, padding=1, stride=1),
            nn.ReLU()
        )
        for i in range(num_cnn_blocks):
            self.cnn_backbone.append(ResidualBlock(hidden_dim=hidden_dim))
        self.cnn_backbone.append(nn.AdaptiveMaxPool2d((1, 1)))

        self.hand_processor = nn.Sequential(
            nn.Linear(21, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

    def forward(self, observation):
        board = observation['board']
        hand = observation['hand']
        vis_feat = self.cnn_backbone(board).flatten(start_dim=1)
        hand_feat = self.hand_processor(hand).flatten(start_dim=1)
        feats = torch.cat(vis_feat, hand_feat)
        return feats


class RandomPolicy(BaseModel):
    def predict(self, obs, mask=None):
        action = self.action_space.sample(mask=np.array(mask, dtype=np.int8))
        return action
