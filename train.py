import argparse
from stable_baselines3.common.env_checker import check_env
from environments.blokus_environment import BlokusEnv
from stable_baselines3.dqn import DQN

def main(n_steps=10):
    env = BlokusEnv()
    model = DQN(policy='MultiInputPolicy', env=env, learning_rate=1e-4, batch_size=256)
    # check_env(env)
    model.learn(100, progress_bar=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('')
    args = parser.parse_args()
    main()
