from algorithms.a2c import a2c
from algorithms.dqn import dqn
from algorithms.ppo import ppo
from algorithms.q_learning import q_learning
from env.env_wrapper import EmptyEnvWrapper, StateRepresentation
from env.env_wrapper import KeyEnvWrapper
from env.key_flat_obs_wrapper import KeyFlatObsWrapper
from env.random_empty_env_10 import RandomEmptyEnv_10
from env.random_key_env_10 import RandomKeyMEnv_10
from util.reports_util import log_training_process, create_report

import os
import time
from experiment_config import *

NA = 'N/A'

ALGORITHM_IMPL = {
    Algorithm.QL: q_learning,
    Algorithm.DQN: dqn,
    Algorithm.PPO: ppo,
    Algorithm.A2C: a2c
}

def create_env(env_type, state_representation, env_stochasticity):
    if env_type == EnvType.EMPTY:
        source_env = KeyFlatObsWrapper(RandomEmptyEnv_10(render_mode='rgb_array'))
        source_env.reset()
        env = EmptyEnvWrapper(source_env, state_representation=state_representation, env_stochasticity=env_stochasticity)
    else:
        source_env = KeyFlatObsWrapper(RandomKeyMEnv_10(render_mode='rgb_array'))
        source_env.reset()
        env = KeyEnvWrapper(source_env, state_representation=state_representation, env_stochasticity=env_stochasticity)
    return env

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
    f.write(f'{tested_parameter},done_episodes_count,total_episodes_count,total_steps_avg,rewards_avg,loss_avg\n')
    f.close()

    return train_result_file, test_result_file

def evaluate_policy(env, policy, steps_cutoff):
    '''
    Evaluates a given policy on the given env for a limited steps_cutoff number of steps or until the environment is solved.
    :return: A tuple containing two values:
        - steps_count: The number of steps taken during the evaluation, capped at steps_cutoff.
        - done: A boolean indicating whether the environment was solved during the evaluation.
    :rtype: tuple
    '''
    state = env.reset()
    done = False
    steps_count = 0

    while not done and steps_count < steps_cutoff:
        action = policy(state)
        state, reward, done = env.step(action)
        steps_count += 1

    return steps_count, done

def run_experiment(env, env_params, algorithm, algorithm_params, tested_parameter, tested_values):
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

        env.set_params(**env_params_cpy)
        env.reset()

        print(f'Evaluating parameter: {tested_parameter}={parameter_value}.')

        #################
        # Step 1: Train #
        #################
        print("Start training")

        algorithm_params_cpy['env'] = env
        mid_train_policy, policy, states_visits_mean, done_count, episodes_steps, episodes_rewards, *episodes_loss = algorithm(**algorithm_params_cpy)
        episodes_loss = episodes_loss[0] if episodes_loss else [] # tabular algorithms do not provide loss information - this is relevant only for deep

        # First - log training process
        parameter_train_log_path = f'{train_log_path}/{tested_parameter}_{parameter_value}'  # Create training log path
        if not os.path.exists(parameter_train_log_path):
            os.makedirs(parameter_train_log_path)
        log_training_process(parameter_train_log_path, states_visits_mean, episodes_steps, episodes_rewards, episodes_loss)

        # Then - aggregate training results and log
        f = open(train_result_file, 'a')
        total_steps_avg = np.mean(episodes_steps)
        rewards_avg = np.mean(episodes_rewards)
        loss_avg = np.mean(episodes_loss) if episodes_loss else NA
        f.write(f'{parameter_value},{done_count},{algorithm_params_cpy["num_episodes"]},{total_steps_avg},{rewards_avg},{loss_avg}\n')
        f.close()

        ################
        # Step 2: Test #
        ################
        print('Start testing')
        steps_count, done = evaluate_policy(env, policy, steps_cutoff=test_steps_cutoff)

        f = open(test_result_file, 'a')
        f.write(f'{parameter_value},{done},{steps_count}\n')
        f.close()

        create_report(result_path, tested_parameter, train_result_file, test_result_file, train_log_path)

if __name__ == '__main__':
    start_time = time.time()
    state_representation = StateRepresentation.ENCODED if Algorithm.QL == algo_type else StateRepresentation.IMAGE
    env = create_env(env_type, state_representation, env_stochasticity)

    algorithm = ALGORITHM_IMPL[algo_type]
    algorithm_params = algorithms_params[algo_type]

    tested_parameters_dict = {**tested_parameters[ENV_PARAMS], **tested_parameters[algo_type]}

    for tested_parameter, tested_values in tested_parameters_dict.items():
        run_experiment(env, env_params, algorithm, algorithm_params, tested_parameter, tested_values)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Total time to run: {execution_time} seconds.')