import argparse

import numpy as np
from gym.vector.utils import spaces

import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_util import Monitor

from environments.SelfPlayBlokusEnv import SelfPlayBlokusEnv
from environments.SingleplayerBlokusEnv import SingleplayerBlokusEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from policy import RandomPolicyDiscrete, RandomPolicyMultiDiscrete, BlokusSeer
from wandb.integration.sb3 import WandbCallback


def get_policy_kwargs(feature_extractor):
    return {
        'BlokusSeer': {'features_extractor_class': BlokusSeer},
        'default': {}
    }[feature_extractor]


def get_random_policy(action_mode):
    if action_mode == 'discrete_masked':
        player = RandomPolicyDiscrete(
            action_space=spaces.Discrete(67200),
            observation_space=spaces.Discrete(67200)
        )
    elif action_mode == 'multi_discrete':
        player = RandomPolicyMultiDiscrete(
            action_space=spaces.MultiDiscrete([400, 21, 8]),
            observation_space=spaces.MultiDiscrete([400, 21, 8])
        )
    return player


def main(envs=8, n_steps=50000, batch_size=256, lr=1e-4, feature_extractor='default', exp_name='test', wandb_log=False,
         action_mode='multi_discrete', env_mode='singleplayer'):
    # action_mode can be 'discrete_masked' or 'multi_discrete'
    # env_mode can be 'self_play' or 'singleplayer'

    env_kwargs = {
        'render_mode': None,
        'action_mode': action_mode,
        'd_board': 20  # 20 default, how about 10 instead for singleplayer?
    }

    if env_mode == 'self_play':
        # SELF PLAY
        player = get_random_policy(action_mode)
        if action_mode == 'discrete_masked':
            compet = 'model'
        else:
            compet = 'random'

        if envs > 0:
            env = SubprocVecEnv(env_fns=[lambda: Monitor(SelfPlayBlokusEnv(
                p2=player, p3=player, p4=player, competitors=compet, **env_kwargs))] * envs)
        else:
            env = Monitor(SelfPlayBlokusEnv(p2=player, p3=player,
                                            p4=player, competitors=compet, **env_kwargs))

    elif env_mode == 'singleplayer':
        # SINGLEPLAYER
        if envs > 0:
            env = SubprocVecEnv(env_fns=[lambda: Monitor(
                SingleplayerBlokusEnv(**env_kwargs))] * envs)
        else:
            env = Monitor(SingleplayerBlokusEnv(**env_kwargs))

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
    env_kwargs['render_mode'] = 'human'

    if env_mode == 'self_play':
        # SELF PLAY
        env = Monitor(SelfPlayBlokusEnv(
            p2=player, p3=player, p4=player, **env_kwargs))
    elif env_mode == 'singleplayer':
        # SINGLEPLAYER
        env = Monitor(SingleplayerBlokusEnv(**env_kwargs))

    rew, l = evaluate_policy(
        model, env, n_eval_episodes=50, render='human', return_episode_rewards=True)
    win_rate = np.mean(np.array(rew) > env.get_wrapper_attr('win_reward'))
    print("AVERAGE REWARD: ", rew, "WIN RATE:", win_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', default=5000000, type=int)
    parser.add_argument('--envs', default=8, type=int)
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--feature_extractor', default='default', type=str)
    parser.add_argument('--exp_name', default='test', type=str)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--act_mode', default='multi_discrete', type=str)
    parser.add_argument('--env_mode', default='singleplayer', type=str)
    args = parser.parse_args()

    if args.wandb:
        wandb.init(name=args.exp_name,
                   config=vars(args),
                   entity='blokusrl',
                   project='blokusrl',
                   sync_tensorboard=True)

    main(n_steps=args.n_steps, envs=args.envs, batch_size=args.bs,
         lr=args.lr, exp_name=args.exp_name, feature_extractor=args.feature_extractor,
         wandb_log=args.wandb, action_mode=args.act_mode, env_mode=args.env_mode)
