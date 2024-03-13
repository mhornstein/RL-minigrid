import gym
import copy
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from enum import Enum

agent_directions_space = 4
key_state_space = 2
door_state_space = 2
class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class KeyEnvWrapper(gym.Env):
    '''
    This wrapper enables environment customization
    '''
    def __init__(self, env, step_reward=-0.01, goal_reward=10):
        '''
        Initializes the EnvWrapper.

        Args:
            env (gym.Env): The Gym environment to be wrapped. Important: this env must be initialized before wrapping (use env.reset())
            step_reward (float, optional): The per-step reward or penalty. Default is -0.01.
            goal_reward (float, optional): The reward for reaching the goal. Default is 10.
        '''
        self.source_env = env
        self.reset()

        self.set_params(step_reward, goal_reward)

    def reset(self): # open gym default implementation changes the board. override, as we do not want to change the board upon reset every time
        self.env = copy.deepcopy(self.source_env)
        self.was_key_picked_up = False
        self.was_door_unlocked = False
        return self.get_current_state()

    def set_params(self, step_reward, goal_reward):
        self.step_reward = step_reward
        self.goal_reward = goal_reward

    def get_current_state(self):
        '''
        The state holds the knowledge of the agent: the agent knows where it is and what its direction is,
        as well as some history of its actions: whether it has picked up the key or unlocked the door for the first time
        '''
        agent_col, agent_row = self.env.get_position()
        agent_direction = self.env.get_direction()
        state = (agent_col - 1, agent_row - 1, agent_direction, int(self.was_key_picked_up), int(self.was_door_unlocked))
        return state

    def render(self):
        plt.imshow(self.env.render())

    def get_state_dim(self):
        rows = self.env.height - 2
        cols = self.env.width - 2
        return cols, rows, agent_directions_space, key_state_space, door_state_space # add 2 more for key carring and door opening

    def get_action_dim(self):
        return self.env.action_space.n

    def sample_action(self):
        available_actions = [0, 1]

        key_pos = self.env.get_k_pos()
        door_pos = self.env.get_d_pos()

        if self.can_use_element(key_pos): # the agent is fronting a key - if so, we can use pick up action (3)
            available_actions.append(3)
        elif (not self.env.is_door_open() and
              self.env.is_carrying_key() and
              self.can_use_element(door_pos)): # the agent is fronting a door it can open - if so, we can use toggle action (5)
            available_actions.append(5)
        elif not self.env.is_wall_front_pos(): # the agent is not standing in front of a wall - if so, we can proceed forward (2)
            available_actions.append(2)

        sampled_action = random.choice(available_actions)
        return sampled_action

    def can_use_element(self, element_pos):
        agent_direction = self.env.get_direction()
        agent_col, agent_row = self.env.get_position()

        if Direction.UP.value == agent_direction:
            required_pos = agent_col, agent_row - 1
        elif Direction.DOWN.value == agent_direction:
            required_pos = agent_col, agent_row + 1
        elif Direction.LEFT.value == agent_direction:
            required_pos = agent_col - 1, agent_row
        else: # Direction.RIGHT.value == agent_direction
            required_pos = agent_col + 1, agent_row
        return required_pos == element_pos

    def step(self, action):
        # First - log current state
        agent_had_key = self.env.is_carrying_key()
        door_was_opened = self.env.is_door_open()

        # now - perform the action and check its outcome
        _ = self.env.step(action)
        agent_has_key = self.env.is_carrying_key()
        door_is_opened = self.env.is_door_open()
        done = self.env.get_goal_pos() == self.env.get_position()

        if done:
            r = self.goal_reward
        elif not agent_had_key and agent_has_key and not self.was_key_picked_up:  # the agent picked up the key for the first time
            self.was_key_picked_up = True
            r = 1 # TODO remove to constant
        elif not door_was_opened and door_is_opened and not self.was_door_unlocked: # the door is opened for the first time
            self.was_door_unlocked = True
            r = 1 # TODO remove to constant
        else: # this is a regular step
            r = self.step_reward

        s_tag = self.get_current_state()

        return s_tag, r, done

# Testing
if __name__ == '__main__':
    from common.random_key_env_10 import RandomKeyMEnv_10
    from common.key_flat_obs_wrapper import KeyFlatObsWrapper
    import matplotlib.pyplot as plt

    source_env = KeyFlatObsWrapper(RandomKeyMEnv_10(render_mode='rgb_array'))
    source_env.reset()  # we must reset the env: this places the agent and other elements on top of the board
    env = KeyEnvWrapper(source_env)

    state = env.reset()
    print(f'start_state: {state}')

    print(f'states dim: {env.get_state_dim()}')
    print(f'action dim: {env.get_action_dim()}')

    action = env.sample_action()
    print(f'sampled action: {action}')
    s_tag, r, done = env.step(action)