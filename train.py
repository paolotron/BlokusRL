import argparse
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import Monitor
from environments.blokus_environment import BlokusEnv
from stable_baselines3.dqn import DQN


def main(n_steps=10):
    env = BlokusEnv()
    model = DQN(policy='MultiInputPolicy', env=env, learning_rate=1e-4, batch_size=256, verbose=True)
    model.learn(100000, progress_bar=True)
    env = Monitor(BlokusEnv(render_mode='human'))
    rew, l = evaluate_policy(model, env, render=True)
    print("AVERAGE REWARD: ", rew)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('')
    args = parser.parse_args()
    main()
