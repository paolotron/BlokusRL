import argparse

import numpy as np
from gym.vector.utils import spaces

import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_util import Monitor

from environments.SelfPlayBlokusEnv import SelfPlayBlokusEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from environments.blokus_environment import BlokusEnv
from policy import RandomPolicy, BlokusSeer
from wandb.integration.sb3 import WandbCallback


def get_policy_kwargs(feature_extractor):
    return {
        'BlokusSeer': {'features_extractor_class': BlokusSeer},
        'default': {}
    }[feature_extractor]


def get_random_policy():
    player = RandomPolicy(action_space=spaces.Discrete(67200), observation_space=spaces.Discrete(67200))
    return player


def main(envs=8, n_steps=10, batch_size=256, lr=1e-4, feature_extractor='default', exp_name='test', wandb_log=False):
    player = get_random_policy()
    env_kwargs = {
        'dummy_competitor': False,
        'render_mode': None
    }
    if envs > 0:
        env = SubprocVecEnv(env_fns=[lambda: Monitor(SelfPlayBlokusEnv(p2=player, p3=player, p4=player, **env_kwargs))] * envs)
    else:
        env = Monitor(SelfPlayBlokusEnv(p2=player, p3=player, p4=player, **env_kwargs))

    model = MaskablePPO(
        policy='MultiInputPolicy',
        policy_kwargs=get_policy_kwargs(feature_extractor),
        n_steps=100,
        env=env,
        learning_rate=lr,
        batch_size=batch_size,
        verbose=1,
        tensorboard_log=f'./log/{exp_name}'
    )
    
    if wandb_log:
        model.learn(
            total_timesteps=n_steps,
            callback=WandbCallback(log='all', verbose=2),
            log_interval=1

        )
    else:
        model.learn(total_timesteps=n_steps)
    
    model.save(f'./logs/{exp_name}')
    env.close()
    env = Monitor(SelfPlayBlokusEnv(p2=player, p3=player, p4=player, render_mode=None))
    rew, l = evaluate_policy(model, env, n_eval_episodes=50, render=False, return_episode_rewards=True)
    win_rate = np.mean(np.array(rew) > env.get_wrapper_attr('win_reward'))
    print("AVERAGE REWARD: ", rew, "WIN RATE:", win_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', default=50000, type=int)
    parser.add_argument('--envs', default=4, type=int)
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--feature_extractor', default='default', type=str)
    parser.add_argument('--exp_name', default='test', type=str)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(name=args.exp_name,
                   config=vars(args),
                   entity='blokusrl',
                   project='blokusrl',
                   sync_tensorboard=True)

    main(n_steps=args.n_steps, envs=args.envs, batch_size=args.bs,
         lr=args.lr, exp_name=args.exp_name, feature_extractor=args.feature_extractor,
         wandb_log=args.wandb)
