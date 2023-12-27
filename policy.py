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
        y = self.relu(y)
        return y + x


class BlokusSeer(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, num_cnn_blocks=3, features_dim=128):
        super().__init__(observation_space, features_dim=features_dim)
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(4, features_dim // 2, 3, padding=1, stride=1),
            nn.ReLU()
        )
        for i in range(num_cnn_blocks):
            self.cnn_backbone.append(ResidualBlock(hidden_dim=features_dim//2))
        self.cnn_backbone.append(nn.AdaptiveMaxPool2d((1, 1)))

        self.hand_processor = nn.Sequential(
            nn.Linear(21 * 4, features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim // 2),
            nn.ReLU(),
        )

    def forward(self, observation):
        board = observation['board'].permute(0, 3, 1, 2)
        hand = observation['hands']
        vis_feat = self.cnn_backbone(board).flatten(start_dim=1)
        hand_feat = self.hand_processor(hand.flatten(start_dim=1)).flatten(start_dim=1)
        feats = torch.concatenate([vis_feat, hand_feat], dim=1)
        return feats


class RandomPolicyDiscrete(BaseModel):
    def predict(self, obs, mask=None):
        action = self.action_space.sample(mask=np.array(mask, dtype=np.int8))
        return action

class RandomPolicyMultiDiscrete(BaseModel):
    def predict(self, obs, mask=None):
        action = self.action_space.sample(mask=mask)
        return action

