import gym
import torch

from stable_baselines3 import A2C, PPO
from legacy.ppo import PPO
from policies import FeedForwardNN
from legacy.envs_graph import SupplyChain


def main():
    hyperparameters = {
                'timesteps_per_batch': 2048, 
                'max_timesteps_per_episode': 200, 
                'gamma': 0.99, 
                'n_updates_per_iteration': 10,
                'lr': 3e-4, 
                'clip': 0.2
                }
    env = SupplyChain(setup_name='test', observation_type='global_vector')

    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)
    model.learn(total_timesteps=100000)


if __name__ == '__main__':
	main()