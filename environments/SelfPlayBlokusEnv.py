from typing import List, Tuple

from environments.blokus_environment import BlokusEnv
from sb3_contrib.common.maskable.policies import BasePolicy


class SelfPlayBlokusEnv(BlokusEnv):

    def __init__(self, p2: BasePolicy, p3: BasePolicy, p4: BasePolicy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2, self.p3, self.p4 = p2, p3, p4

    def step(self, action):
        obs, reward, _, _, _ = super().step(action)
        action = self.p2.predict(obs)
        obs, _, _, _, _ = super().step(action)
        action = self.p3.predict(obs)
        obs, _, _, _, _ = super().step(action)
        action = self.p4.predict(obs)
        obs, _, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

