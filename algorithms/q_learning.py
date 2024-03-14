import numpy as np

def create_policy(Q):
    q_snapshot = Q.copy()
    def policy(state): return np.argmax(q_snapshot[state][:])  # return a greedy policy
    return policy

def q_learning(env, alpha, gamma, epsilon, ep_decay, num_episodes, steps_cutoff):
    '''
    Q-learning algorithm.

    Params relevant to the env:
    :param env: the OpenAI env wrapped with a Wrapper object

    Params relevant to the algorithm:
    :param alpha: learning rate
    :param gamma: discount factor
    :param epsilon: exploration rate
    :param ep_decay: decaying epsilon value
    :param num_episodes: number of episodes for training
    :param steps_cutoff: maximal steps for episode

    :return: learnt policy (at mid and end of training) + stats monitoring the training process: states_visits_mean, done_counter, episodes_steps (count), episodes_rewards (sum)
    '''

    # Create Q table of size |S| x |A| where:
    # S = varying state size, with columns and rows being the first 2 dimentions:
    # e.g. for empty-env: (agent col, agent row, agent direction) and for key-env (agent col, agent row, agent direction, is_carrying_key, is_door_unlocked)
    # A is in the set {turn right, turn left, move forward}

    # We initialize values in the table, Q(terminal state) = 0
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    Q = np.zeros(state_dim + (action_dim,), dtype=np.float64)

    cols, rows = env.get_board_dims()
    states_visits_count = np.zeros([cols, rows], dtype=float) # Initialize stats' data structures

    done_count = 0
    episodes_steps = []
    episodes_rewards = []

    for ep in range(1, num_episodes+1):
        #print(f'start ep: {ep}')
        s = env.reset()
        states_visits_count[s[0], s[1]] += 1

        done = False
        steps_count = 0
        reward_sum = 0

        # env.render()

        while not done and steps_count < steps_cutoff:
            # Choose an action a based on current policy (e.g. 𝜀 − 𝑔𝑟𝑒𝑒𝑑𝑦)
            if np.random.uniform(0, 1) < epsilon:
                a = env.sample_action()
            else:
                a = np.argmax(Q[s][:])

            # You get a reward r. You are now in state s’
            s_tag, r, done = env.step(a)

            # env.render()

            # Choose an action a’ from s’ based on current policy.
            q_value = np.max(Q[s_tag][:])
            Q[s][a] = Q[s][a] + alpha * (r + gamma * q_value - Q[s][a])

            if s[0:2] != s_tag[0:2]: # if the agent moved to a new location on the board
                states_visits_count[s[0], s[1]] += 1
            reward_sum += r
            steps_count += 1

            s = s_tag

        epsilon = ep_decay * epsilon # Decay exploration rate

        if done:
            done_count += 1
        episodes_steps.append(steps_count)
        episodes_rewards.append(reward_sum)

        if num_episodes // 2 == ep:
            mid_train_policy = create_policy(Q)

    policy = create_policy(Q)

    return mid_train_policy, policy, states_visits_count / num_episodes, done_count, episodes_steps, episodes_rewards