import torch
import random
import torch.nn as nn
from torch.optim import SGD
from collections import namedtuple, deque

Transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state'))

class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 5),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 5),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(12800, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 13)
        )

    def forward(self, x):
        return self.model(x)

class ExperienceReplayBuffer:
    def __init__(self, memory_buffer_size):
        self.buffer = deque([], maxlen=memory_buffer_size)
    def push(self, state, action, reward, next_state):
        t = (state, action, reward, next_state)
        self.buffer.append(t)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

def state_to_tensor(state, done=False):
    if done:
        return None
    else:
        return torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

def create_policy(policy_net):
    policy_net_snapshot = QNN()
    policy_net_snapshot.load_state_dict(policy_net.state_dict())
    def policy(s):
        '''
        This function gets a state and returns the preferable action.
        It does it by providing the state to the network and returning the action with maximal q-value
        (i.e. this is a greedy policy_net)
        '''
        state_tensor = state_to_tensor(s)
        with torch.no_grad():
            q_values = policy_net_snapshot(state_tensor)
        a = q_values.max(1)[1].view(1, 1)
        return a.item()
    return policy

def pick_action(epsilon, state, env, policy_net):
    '''
    Selects an action based on the epsilon-greedy strategy, which balances exploration and exploitation.
    '''
    if random.random() >= epsilon: # exploitation
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else: # exploration
        return torch.tensor([[env.sample_action()]], dtype=torch.long)

def update_target_net(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())

def update_epsilon(epsilon, ep_decay):
    ep = max(epsilon * ep_decay, 0.05)
    return ep

def train_policy_network(buffer, policy_net, target_net, batch_size, gamma, optimizer, criterion):
    # Step 1: sample from data and create the required tensors
    batch = buffer.sample(batch_size)
    state_tensor = torch.cat(batch.state)
    action_tensor = torch.cat(batch.action)
    reward_tensor = torch.cat(batch.reward)
    next_state_tensor = torch.cat([next_state for next_state in batch.next_state if next_state is not None])
    non_final_state_mask_tensor = torch.tensor(tuple(map(lambda next_state: next_state is not None, batch.next_state)), dtype=torch.bool)

    # Step 2: Calculate Q-values for current states and selected actions
    q_values_tensor = policy_net(state_tensor).gather(1, action_tensor)

    # Step 3: Calculate the maximum Q-values for the next states using the target network
    next_q_values_tensor = torch.zeros(batch_size)
    with torch.no_grad():
        next_q_values_tensor[non_final_state_mask_tensor] = target_net(next_state_tensor).max(1)[0]  # Note: final states remains with reward of 0
    target_q_values = (next_q_values_tensor * gamma) + reward_tensor

    # Step 4: Calculate the loss and update the model accordingly
    loss = criterion(q_values_tensor, target_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def dqn(env, num_episodes, batch_size, gamma, ep_decay, epsilon,
        target_freq_update, memory_buffer_size, learning_rate, steps_cutoff, train_action_value_freq_update):
    policy_net = QNN()
    target_net = QNN()
    update_target_net(policy_net, target_net)
    criterion = torch.nn.MSELoss()

    buffer = ExperienceReplayBuffer(memory_buffer_size)
    optimizer = SGD(policy_net.parameters(), lr=learning_rate)

    done_count = 0
    episodes_loss, episodes_rewards, episodes_steps = [], [], []

    for i in range(1, num_episodes+1):
        print(f'Running episode {i}\\{num_episodes}\nPrinting steps progress (up to {steps_cutoff} steps): ')

        done = False
        episode_reward = 0
        episode_loss = 0
        num_steps = 1

        state = state_to_tensor(env.reset())

        while not done and num_steps <= steps_cutoff:
            print(num_steps, end = " ")
            action = pick_action(epsilon, state, env, policy_net)
            next_state, reward, done, info = env.step(action.item())
            episode_reward += reward

            next_state = state_to_tensor(next_state, done)
            reward = torch.tensor([reward])
            buffer.push(state, action, reward, next_state)

            if len(buffer) >= batch_size and num_steps % train_action_value_freq_update == 0:
                loss = train_policy_network(buffer, policy_net, target_net, batch_size, gamma, optimizer, criterion)
                episode_loss += loss

            num_steps += 1
            epsilon = update_epsilon(epsilon, ep_decay)
            state = next_state

        num_steps = steps_cutoff if not done else num_steps

        episodes_rewards.append(episode_reward)
        episodes_loss.append(episode_loss / num_steps)
        episodes_steps.append(num_steps)

        if i % target_freq_update == 0:
            update_target_net(policy_net, target_net)

        if done:
            done_count += 1

        if num_episodes // 2 == i:
            mid_train_policy = create_policy(policy_net)

        print()

    policy = create_policy(policy_net)

    return mid_train_policy, policy, policy_net, done_count, episodes_steps, episodes_rewards, episodes_loss