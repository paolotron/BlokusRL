import argparse

import numpy as np

from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_util import Monitor
from sb3_contrib import MaskablePPO
from environments.SelfPlayBlokusEnv import SelfPlayBlokusEnv
from policy import BlokusSeer


def get_policy_kwargs(feature_extractor):
    return {
        'BlokusSeer': {'features_extractor_class': BlokusSeer},
        'default': {}
    }[feature_extractor]


def main(exp_name='test', feature_extractor='default'):
    
    env_kwargs = {
        'render_mode': 'human',
        'action_mode': 'multi_discrete'
    }
    player = None
    env = Monitor(SelfPlayBlokusEnv(p2=player, p3=player, p4=player, **env_kwargs))
    model = MaskablePPO(
        policy='MultiInputPolicy',
        policy_kwargs=get_policy_kwargs(feature_extractor),
        env=env,
    )
    model = model.load(f'./logs/{exp_name}')
    rew, l = evaluate_policy(model, env, n_eval_episodes=50, render='human', return_episode_rewards=True)
    win_rate = np.mean(np.array(rew) > env.get_wrapper_attr('win_reward'))
    print("AVERAGE REWARD: ", rew, "WIN RATE:", win_rate)
    

if __name__ == '__main__':
    main()