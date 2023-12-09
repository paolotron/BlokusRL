from typing import List, Tuple

from environments.blokus_environment import BlokusEnv
from stable_baselines3.common.policies import BaseModel


class SelfPlayBlokusEnv(BlokusEnv):

    def __init__(self, p2: BaseModel, p3: BaseModel, p4: BaseModel, dummy_competitor=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2, self.p3, self.p4 = p2, p3, p4
        self.dummy_random = dummy_competitor

    def step(self, action):
        obs, reward, _, _, _ = super().step(action)
        action = self.p2.predict(obs, mask=self.action_masks())
        obs, _, _, _, _ = super().step(action)
        action = self.p3.predict(obs, mask=self.action_masks())
        obs, _, _, _, _ = super().step(action)
        action = self.p4.predict(obs, mask=self.action_masks())
        obs, _, terminated, truncated, info = super().step(action)
        if self.dead[0]:
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        output = super().reset(seed)
        if self.dummy_random:
            seed = 0
        self.p2.action_space.seed(seed=seed)
        self.p3.action_space.seed(seed=seed)
        self.p4.action_space.seed(seed=seed)
        return output
