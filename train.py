import argparse

from gym.vector.utils import spaces
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.env_util import Monitor
from environments.blokus_environment import BlokusEnv
from environments.SelfPlayBlokusEnv import SelfPlayBlokusEnv
from stable_baselines3.dqn import DQN

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from policy import RandomPolicy


def main(n_steps=10):
    dummy_env = BlokusEnv()
    player = RandomPolicy(action_space=dummy_env.action_space, observation_space=dummy_env.observation_space)
    env = SelfPlayBlokusEnv(p2=player, p3=player, p4=player)

    model = MaskablePPO(policy='MultiInputPolicy',
                        n_steps=100,
                        env=env,
                        learning_rate=1e-4,
                        batch_size=256,
                        verbose=True)

    model.learn(total_timesteps=5000, progress_bar=True)
    env = Monitor(SelfPlayBlokusEnv(p2=player, p3=player, p4=player, render_mode='human'))
    rew, l = evaluate_policy(model, env, render=True)
    print("AVERAGE REWARD: ", rew)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('')
    args = parser.parse_args()
    main()
