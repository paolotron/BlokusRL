from typing import List, Tuple

from environments.blokus_environment import BlokusEnv
from stable_baselines3.common.policies import BaseModel


class SelfPlayBlokusEnv(BlokusEnv):

    def __init__(self, p2: BaseModel, p3: BaseModel, p4: BaseModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2, self.p3, self.p4 = p2, p3, p4

    def step(self, action):
        obs, reward, _, _, _ = super().step(action)
        action = self.p2.predict(obs, mask=self.action_masks())
        obs, _, _, _, _ = super().step(action)
        action = self.p3.predict(obs, mask=self.action_masks())
        obs, _, _, _, _ = super().step(action)
        action = self.p4.predict(obs, mask=self.action_masks())
        obs, _, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

