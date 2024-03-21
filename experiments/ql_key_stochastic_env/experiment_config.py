from enum import Enum

import numpy as np

import random
random.seed(42) # we want to set a seed so our environment will be the same across experiments

class EnvType(Enum):
    EMPTY = 0
    KEY = 1

class EnvStochasticity(Enum):
    CONSTANT = 0
    STOCHASTIC = 1

class Algorithm(Enum):
    QL = 'QL'
    DQN = 'DQN'

ENV_PARAMS = 'Env_params'

#############################
# Experiments configuration #
#############################
'''
Training-phase parameters:
train_num_episodes - the number of episodes for the training
train_steps_cutoff - maximal steps allowed per episode
'''
train_num_episodes = (2*8-1)*8*300
train_steps_cutoff = 100

'''
Evaluation-phase parameters
test_num_episodes - the number of episodes for the evaluation
test_steps_cutoff - maximal steps allowed per episode
'''
test_steps_cutoff = 100

'''
Default hyper-parameters values for the algorithms.
The tested hyperparameter will override its value in the relevant test
'''
algorithms_params = {
    Algorithm.QL: {
        'alpha': 0.999,
        'gamma': 1,
        'epsilon': 1.0,
        'ep_decay': 0.9995,
        'num_episodes': train_num_episodes,
        'steps_cutoff': train_steps_cutoff
    },
    Algorithm.DQN: {
        'learning_rate': 0.0001,
        'gamma': 0.999,
        'epsilon': 1,
        'ep_decay': 0.999,
        'num_episodes': train_num_episodes,
        'steps_cutoff': train_steps_cutoff,
        'batch_size': 32,
        'target_freq_update': 10,
        'memory_buffer_size': 10000,
        'train_action_value_freq_update': 1
    }
}

'''
Default hyper-parameters values for env
The tested hyperparameter will override its value in the relevant test
'''
env_params = {'goal_reward': 10, 'step_reward': -0.01}

#######################################
# Tested hyperparameter configuration #
#######################################
# remove dictionary keys to test less hyperparameters
# Change the values in the entries to test different hyper-parameters values
tested_parameters = {
    Algorithm.QL: {
        'alpha': [0.9, 0.95, 0.99, 0.999, 1],
        'gamma': [0.8, 0.85, 0.9, 0.95, 1.0],
        'ep_decay': [0.999, 0.9999, 0.99999],
    },
    Algorithm.DQN: {
        'learning_rate': [0.00005, 0.0001, 0.0005, 0.001],
        'batch_size': [16, 32, 64, 128, 256],
        'target_freq_update': [4, 10, 16],
        'memory_buffer_size': [100, 500, 1000, 10000],
        'gamma': [0.9, 0.95, 0.999],
        'ep_decay': [0.9, 0.95, 0.999],
        'epsilon': [0.7, 0.9, 1.0],
        'train_action_value_freq_update': [1, 4, 8, 16]
    },
    ENV_PARAMS: {
        'goal_reward': [-10, 0, 1, 5, 10],
        'step_reward': np.arange(-0.006, 0.005, 0.002).round(3),
    }
}

# Pick the env stochasiticity you want: use EnvStochasticity.CONSTANT to keep artifacts in their place upon reset, or use STOCHASTIC to allow them to change upon reset
env_stochasticity = EnvStochasticity.STOCHASTIC

# Pick the env you want to test: EnvType.EMPTY or EnvType.KEY
env_type = EnvType.KEY

# Pick the algorithm you want to test: Algorithm.QL or Algorithm.DQL
algo_type = Algorithm.QL