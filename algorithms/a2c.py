from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np
import copy

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class SequentialExperienceReplayBuffer:
    '''
    This buffer supports the A2C algorithm by maintaining experiences in sequential order for on-policy learning.
    Unlike traditional replay buffers that randomly sample past experiences, this buffer ensures experiences are
    used once in the order they were collected, aligning with A2C's requirement for current, policy-generated data.
    After training on these experiences, they are discarded, enabling batch updates while adhering to on-policy
    principles. This method provides a balance between efficient learning and the on-policy nature of A2C,
    facilitating stable gradient estimates and potentially faster convergence.
    '''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def get_sequential(self):
        while self.buffer:
            yield self.buffer.popleft()

    def reset(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

def state_to_tensor(state, done=False): # same as dqn
    if done:
        # This tensor serves as a placeholder representing the next state when the agent
        # reaches a terminal state (done=True). In the neural network training process,
        # the evaluation of next state values is not considered for backpropagation
        # calculations. Therefore, using this placeholder does not impact the accuracy
        # of the network's updates. The value predicted will be zeroed-out by the loss calcualtion.
        return torch.zeros((1, 3, 320, 320), dtype=torch.float32)
    else:
        return torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

def create_policy(policy_net):
    policy_net_snapshot = copy.deepcopy(policy_net)
    policy_net_snapshot.eval()

    def policy(state):
        state_tensor = state_to_tensor(state)
        action_probs, state_value = policy_net(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample().item()
        return action

    return policy


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

        self.fc = nn.Linear(41472, 512)
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = F.relu(self.fc(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value

def update_a2c(policy_net, optimizer, transitions, gamma, entropy_weight):
    """
    Updates the actor-critic network.

    Args:
    - policy_net: The actor-critic network.
    - optimizer: The optimizer.
    - transitions: A batch of experience transitions.
    - gamma: Discount factor for future rewards.
    - entropy_weight: The weight given to the entropy term in the loss calculation, to balance exploration and exploitation.

    Returns:
    - loss: The computed loss for this batch of transitions.
    """
    # Unpack the transitions
    states, actions, rewards, next_states, dones = zip(*transitions)

    # Convert lists to PyTorch tensors
    states = torch.cat(states)
    actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute values for the current and next states
    action_probs, current_values = policy_net(states)
    with torch.no_grad():
        _, next_values = policy_net(next_states)

    # Compute the expected values (targets) for the current state values
    targets = rewards + gamma * next_values.squeeze() * (1 - dones)

    # Compute the advantages
    advantages = targets - current_values.squeeze()

    # Compute log probabilities of the taken actions
    dist = torch.distributions.Categorical(action_probs)
    log_probs = dist.log_prob(actions.squeeze())

    # Actor loss (policy gradient)
    actor_loss = -(log_probs * advantages.detach()).mean()

    # Critic loss (value function loss)
    critic_loss = F.mse_loss(current_values.squeeze(), targets.detach())

    # Adding entropy to penalize too much exploitation
    entropy = dist.entropy().mean()

    # Total loss
    loss = actor_loss + critic_loss - entropy_weight * entropy

    # Perform a gradient update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def a2c(env, num_episodes, steps_cutoff, gamma, learning_rate, train_freq, entropy_weight):
    action_dim = env.get_action_dim()
    policy_net = CNNActorCritic(action_dim)
    optimizer = Adam(policy_net.parameters(), lr=learning_rate)
    buffer = SequentialExperienceReplayBuffer(train_freq)

    cols, rows = env.get_board_dims()
    states_visits_count = np.zeros([cols, rows], dtype=float) # Initialize stats' data structures

    done_count = 0
    episodes_loss, episodes_rewards, episodes_steps = [], [], []

    for i in range(1, num_episodes+1):
        print(f'\nRunning episode {i}\\{num_episodes}\nPrinting steps progress (up to {steps_cutoff} steps): ')

        done = False
        episode_reward = 0
        episode_loss = 0
        episode_train_count = 0
        num_steps = 1

        state = state_to_tensor(env.reset())
        agent_position = env.get_agent_position()
        states_visits_count[agent_position] += 1

        while not done and num_steps <= steps_cutoff:
            print(num_steps, end=" ")

            # Step 1: pick an action
            with torch.no_grad():
                action_probs, state_value = policy_net(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            # print(f'{action.item()}', end = ' ')

            next_state, reward, done = env.step(action.item())
            next_state = state_to_tensor(next_state, done)

            # Step 2: update metrics
            episode_reward += reward
            num_steps += 1

            next_agent_position = env.get_agent_position()
            if agent_position != next_agent_position: # if the agent moved to a new location on the board
                states_visits_count[next_agent_position] += 1
            agent_position = next_agent_position

            reward = torch.tensor([reward], dtype=torch.float32)
            done = torch.tensor([done], dtype=torch.float32)

            # Step 3: update memory buffer
            buffer.push(state, action, reward, next_state, done)

            state = next_state

            # Step 4: if possible - train
            if len(buffer) >= train_freq:
                transitions = list(buffer.get_sequential())
                loss = update_a2c(policy_net, optimizer, transitions, gamma, entropy_weight)
                episode_loss += loss
                episode_train_count += 1
                buffer.reset()  # Clear buffer after processing

        num_steps = steps_cutoff if not done else num_steps

        episodes_rewards.append(episode_reward)
        if episode_train_count == 0:
            if len(episodes_loss) == 0:
                episodes_loss.append(0)
            else:
                episodes_loss.append(episodes_loss[-1])
        else:
            episodes_loss.append(episode_loss / episode_train_count)
        episodes_steps.append(num_steps)

        if done:
            done_count += 1
            # print("done! " , end = ' ')

        if num_episodes // 2 == i:
            mid_train_policy = create_policy(policy_net)

    policy = create_policy(policy_net)

    return mid_train_policy, policy, states_visits_count / num_episodes, done_count, episodes_steps, episodes_rewards, episodes_loss