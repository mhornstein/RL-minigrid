import gym
import random
import numpy as np
import copy

agent_directions_space = 4

class EnvWrapper(gym.Env):
    '''
    This wrapper enables environment customization
    '''
    def __init__(self, env):
        env.reset() # we must reset the env: this places the agent and other elements on top of the board
        self.source_env = env
        self.reset()

    def reset(self): # open gym default implementation changes the board. override, as we do not want to change the board upon reset every time
        self.env = copy.deepcopy(self.source_env)
        return self.get_current_state()

    def get_current_state(self):
        agent_col, agent_row = self.env.get_position()
        agent_direction = self.env.get_direction()
        state = (agent_col - 1, agent_row - 1, agent_direction) # remove extra row \ col in the board perimeter
        return state

    def render(self):
        plt.imshow(self.env.render())

    def get_state_dim(self):
        rows = self.env.height - 2
        cols = self.env.width - 2
        return cols, rows, agent_directions_space

    def get_action_dim(self):
        return self.env.action_space.n

    def sample_action(self):
        return self.env.action_space.sample()

# Testing
if __name__ == '__main__':
    from common.RandomKeyMEnv_10 import RandomEmptyEnv_10
    from common.KeyFlatObsWrapper import KeyFlatObsWrapper
    import matplotlib.pyplot as plt

    env = EnvWrapper(KeyFlatObsWrapper(RandomEmptyEnv_10(render_mode='rgb_array')))

    state = env.reset()
    print(f'start_state: {state}')

    print(f'states dim: {env.get_state_dim()}')
    print(f'action dim: {env.get_action_dim()}')

    action = env.sample_action()
    print(f'sampled action: {action}')