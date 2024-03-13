from enum import Enum

import numpy as np

import random
random.seed(42) # we want to set a seed so our environment will be the same across experiments

class EnvType(Enum):
    EMPTY = 0
    KEY = 1

#############################
# Experiments configuration #
#############################
'''
Training-phase parameters:
train_num_episodes - the number of episodes for the training
train_steps_cutoff - maximal steps allowed per episode
'''
train_num_episodes = 500
train_steps_cutoff = 100

'''
Evaluation-phase parameters
test_num_episodes - the number of episodes for the evaluation
test_steps_cutoff - maximal steps allowed per episode
'''
test_steps_cutoff = 100

# Default hyper-parameters values for algorithm (the tested hyperparameter will override its value in the relevant test)
algorithm_params = {'alpha': 0.9, 'gamma': 1, 'epsilon': 1.0, 'ep_decay': 0.99,
                    'num_episodes': train_num_episodes, 'steps_cutoff': train_steps_cutoff}

# Default hyper-parameters values for env (the tested hyperparameter will override its value in the relevant test)
env_params = {'goal_reward': 10, 'step_reward': -0.01}

#######################################
# Tested hyperparameter configuration #
#######################################
# remove dictionary keys to test less hyperparameters
# Change the values in the entries to test different hyper-parameters values
tested_parameters = {'goal_reward': [-10, 0, 1, 5, 10],
                    'step_reward': np.arange(-0.006, 0.007, 0.002).round(3),
                    'alpha': np.arange(0.85, 1.01, 0.025).round(2),
                    'gamma': np.arange(0.88, 1.01, 0.02).round(2),
                    'ep_decay': [0.85, 0.9, 0.95, 0.99, 1]}

env_type = EnvType.EMPTY