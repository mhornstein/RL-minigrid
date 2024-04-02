import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np
import copy

def state_to_tensor(state): # same as dqn, but this time - no need to worry about done states
    return torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

class CNNActorCritic(nn.Module): # same as ppo
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

def create_policy(policy_net): # same as ppo
    policy_net_snapshot = copy.deepcopy(policy_net)
    policy_net_snapshot.eval()

    def policy(state):
        state_tensor = state_to_tensor(state)
        _, action = pick_action(state_tensor, policy_net_snapshot)
        return action

    return policy

def update_a2c(policy_net, optimizer, states, actions, rewards, dones, gamma):
    states = torch.tensor( np.array(states), dtype=torch.float32).permute(0, 3, 1, 2) # convert list first to numpy array for efficiency reasons (Creating a tensor from a list of numpy.ndarrays is extremely slow)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.int)

    # Step 1: get values for all states
    with torch.no_grad():
        _, state_values = policy_net(states)
        state_values = state_values.squeeze()

    # Step 2: calculate the advantages
    # rewards[:-1] and dones[:-1] refer to the evaluated state
    # state_values[1:] refers to the next state
    returns = rewards[:-1] + gamma * state_values[1:] * (1 - dones[:-1])
    advantages = returns - state_values[:-1]

    # Step 4: Compute Policy Loss (the PPO Clip Objective function)
    new_probs, _ = policy_net(states[:-1])
    dist = Categorical(new_probs)
    log_probs = dist.log_prob(actions[:-1])
    policy_loss = -(log_probs * advantages.detach()).mean()

    # Step 5: Computing the value loss (plain old MSE)
    value_loss = F.mse_loss(state_values[:-1], returns.detach())


    # Step 6: combine policy loss and value loss and take gradient step
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def pick_action(state, policy_net):
    state_tensor = state_to_tensor(state)
    with torch.no_grad():
        action_probs, _ = policy_net(state_tensor)
    dist = Categorical(action_probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return log_prob, action

def a2c(env, num_episodes, gamma, lr, update_steps, steps_cutoff):
    action_dim = env.get_action_dim()
    policy_net = CNNActorCritic(action_dim)
    optimizer = Adam(policy_net.parameters(), lr=lr)

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

        state = env.reset()
        agent_position = env.get_agent_position()
        states_visits_count[agent_position] += 1

        states, actions, rewards, dones = [], [], [], []  # We do not need a sophisticated buffer for sampling as we sequentially go over the recent history only - no need for prior history

        while not done and num_steps <= steps_cutoff:
            print(num_steps, end=" ")

            # Step 1: pick an action
            log_prob, action = pick_action(state, policy_net)
            next_state, reward, done = env.step(action.item())

            # Step 2: update metrics
            episode_reward += reward
            num_steps += 1

            next_agent_position = env.get_agent_position()
            if agent_position != next_agent_position:  # if the agent moved to a new location on the board
                states_visits_count[next_agent_position] += 1
            agent_position = next_agent_position

            # Step 3: update memory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state

            # Step 4: if possible - train
            if len(states) >= update_steps + 1 or done:
                loss = update_a2c(policy_net, optimizer, states, actions, rewards, dones, gamma)
                episode_loss += loss
                states, actions, rewards, dones = [states[-1]], [actions[-1]], [rewards[-1]], [dones[-1]] # "clear" the history

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