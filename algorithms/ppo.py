import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from collections import deque, namedtuple
import random
import numpy as np
import copy

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'log_prob', 'done'))

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.buffer)

def state_to_tensor(state, done=False): # same as dqn
    if done:
        return None
    else:
        return torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

class CNNActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(CNNActorCritic, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Linear(41472, 512)  # Adjust this size based on the output of conv_layers
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = F.relu(self.fc(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value

def update_ppo(policy_net, optimizer, batch, gamma, eps_clip):
    states = torch.cat(batch.state)
    actions = torch.stack(batch.action)
    rewards = torch.stack(batch.reward)
    next_states = torch.cat([next_state for next_state in batch.next_state if next_state is not None])
    old_log_probs = torch.stack(batch.log_prob)
    is_terminal = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

    # Get the current policy outputs for old states
    new_probs, state_values = policy_net(states)
    dist = Categorical(new_probs)
    new_log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(1)

    # Calculate the advantages
    with torch.no_grad():
        _, next_state_values = policy_net(next_states)
    returns = rewards + gamma * next_state_values * (1 - is_terminal)
    advantages = (returns - state_values).detach()  # Detach delta to prevent influencing the policy gradient

    # Calculate the ratio (pi_theta / pi_theta_old)
    ratios = torch.exp(new_log_probs - old_log_probs)

    # Compute Policy Loss
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Computing the value loss
    value_loss = F.mse_loss(state_values, returns)

    # Take gradient step
    optimizer.zero_grad()
    loss = policy_loss + 0.5 * value_loss
    loss.backward()
    optimizer.step()

    return loss.item()

def pick_action(state, policy_net):
    with torch.no_grad():
        action_probs, _ = policy_net(state)
    dist = Categorical(action_probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return log_prob, action


def create_policy(policy_net):
    policy_net_snapshot = copy.deepcopy(policy_net)
    policy_net_snapshot.eval()

    def policy(state):
        state_tensor = state_to_tensor(state)
        _, action = pick_action(state_tensor, policy_net_snapshot)
        return action

    return policy


def ppo(env, num_episodes, batch_size, gamma, lr, eps_clip, steps_cutoff, memory_buffer_size, train_freq):
    action_dim = env.get_action_dim()
    policy_net = CNNActorCritic(action_dim)
    optimizer = Adam(policy_net.parameters(), lr=lr)
    buffer = ExperienceReplayBuffer(memory_buffer_size)

    cols, rows = env.get_board_dims()
    states_visits_count = np.zeros([cols, rows], dtype=float) # Initialize stats' data structures

    done_count = 0
    episodes_loss, episodes_rewards, episodes_steps = [], [], []

    for i in range(1, num_episodes+1):
        print(f'\nRunning episode {i}\\{num_episodes}\nPrinting steps progress (up to {steps_cutoff} steps): ')

        done = False
        episode_reward = 0
        episode_loss = 0
        num_steps = 1

        state = state_to_tensor(env.reset())
        agent_position = env.get_agent_position()
        states_visits_count[agent_position] += 1

        while not done and num_steps <= steps_cutoff:
            print(num_steps, end=" ")

            # Step 1: pick an action
            log_prob, action = pick_action(state, policy_net)
            next_state, reward, done = env.step(action.item())

            # Step 2: update metrics
            episode_reward += reward
            num_steps += 1

            next_agent_position = env.get_agent_position()
            if agent_position != next_agent_position: # if the agent moved to a new location on the board
                states_visits_count[next_agent_position] += 1
            agent_position = next_agent_position

            # Step 3: update memory buffer
            reward = torch.tensor([reward])
            done = torch.tensor([done])
            next_state = state_to_tensor(next_state)
            buffer.push(state, action, reward, next_state, log_prob, done)

            state = next_state

            # if possible - train
            if len(buffer) >= batch_size and num_steps % train_freq == 0:
                batch = buffer.sample(batch_size)
                loss = update_ppo(policy_net, optimizer, batch, gamma, eps_clip)
                episode_loss += loss

        num_steps = steps_cutoff if not done else num_steps

        episodes_rewards.append(episode_reward)
        episodes_loss.append(episode_loss / num_steps)
        episodes_steps.append(num_steps)

        if done:
            done_count += 1

        if num_episodes // 2 == i:
            mid_train_policy = create_policy(policy_net)

    policy = create_policy(policy_net)

    return mid_train_policy, policy, states_visits_count / num_episodes, done_count, episodes_steps, episodes_rewards, episodes_loss

if __name__ == '__main__':
    from env.env_wrapper import EmptyEnvWrapper, StateRepresentation
    from env.key_flat_obs_wrapper import KeyFlatObsWrapper
    from env.random_empty_env_10 import RandomEmptyEnv_10
    from experiment_config import EnvStochasticity

    source_env = KeyFlatObsWrapper(RandomEmptyEnv_10(render_mode='rgb_array'))
    source_env.reset()
    env = EmptyEnvWrapper(source_env, state_representation=StateRepresentation.IMAGE, env_stochasticity=EnvStochasticity.CONSTANT)
    ppo(env)