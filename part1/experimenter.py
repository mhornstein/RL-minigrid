import sys

from algorithms.qlearning import q_learning
from common.EmptyEnvWrapper import EnvWrapper

sys.path.append('../')

import os
import time
from experiment_config import *

def init_results_files(tested_parameter, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Test and train stats csv files
    test_result_file = f'{result_path}/test_result_{tested_parameter}.csv'
    f = open(test_result_file, 'w')
    f.write(f'{tested_parameter},solved,steps_count\n')
    f.close()

    train_result_file = f'{result_path}/train_result_{tested_parameter}.csv'
    f = open(train_result_file, 'w')
    f.write(f'{tested_parameter},done_episodes_count,total_episodes_count,total_steps_avg,rewards_avg\n')
    f.close()

    return train_result_file, test_result_file

def run_experiment(env_params, algorithm_params,tested_parameter, tested_values):
    env_params_cpy = env_params.copy()
    algorithm_params_cpy = algorithm_params.copy()
    result_path = f'./results_{tested_parameter}'
    train_result_file, test_result_file = init_results_files(tested_parameter, result_path)

    train_log_path = f'{result_path}/train_log'  # Create training log path
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)

    # Running test for parameter_value
    for parameter_value in tested_values:

        if tested_parameter in env_params_cpy:
            env_params_cpy[tested_parameter] = parameter_value
        else:
            algorithm_params_cpy[tested_parameter] = parameter_value

        env = EnvWrapper(**env_params_cpy)
        algorithm_params_cpy['env'] = env

        print(f'Evaluating parameter: {tested_parameter}={parameter_value}.')

        #################
        # Step 1: Train #
        #################
        print("Start training")
        policy, states_visits_mean, done_count, episodes_steps, episodes_rewards = q_learning(**algorithm_params_cpy)

        # First - log training process
        '''
        parameter_train_log_path = f'{train_log_path}/{tested_parameter}_{parameter_value}'  # Create training log path
        if not os.path.exists(parameter_train_log_path):
            os.makedirs(parameter_train_log_path)
        log_training_process(experiment_log_path, states_visits_mean, episodes_steps, episodes_rewards) # TODO implement
        '''

        # Then - log training results
        f = open(train_result_file, 'a')
        total_steps_avg = np.mean(episodes_steps)
        rewards_avg = np.mean(episodes_rewards)
        f.write(f'{parameter_value},{done_count},{algorithm_params_cpy["num_episodes"]},{total_steps_avg},{rewards_avg}\n')
        f.close()

        ################
        # Step 2: Test #
        ################
        print('Start testing')
        steps_count, done = 0, 0 # evaluate_policy(env, policy, steps_cutoff=steps_cutoff)

        f = open(test_result_file, 'a')
        f.write(f'{parameter_value},{done},{steps_count}\n')
        f.close()

if __name__ == '__main__':
    start_time = time.time()

    for tested_parameter, tested_values in tested_parameters.items():
        run_experiment(env_params, algorithm_params, tested_parameter, tested_values)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Total time to run: {execution_time} seconds.')