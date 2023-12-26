from typing import List, Tuple

from environments.blokus_environment import BlokusEnv
from stable_baselines3.common.policies import BaseModel


class SelfPlayBlokusEnv(BlokusEnv):

    def __init__(self, p2: BaseModel, p3: BaseModel, p4: BaseModel, competitors='random', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2, self.p3, self.p4 = p2, p3, p4
        assert competitors in ('model', 'random')
        self.random_competitor = competitors.startswith('random')

    def step(self, action):
        obs, reward, terminated, truncated, _ = super().step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
        if self.random_competitor:
            for _ in range(3):
                super().random_step()
        else:
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
        return output
