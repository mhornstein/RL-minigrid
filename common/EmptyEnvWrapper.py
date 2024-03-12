import gym
import random
import numpy as np
import copy

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
        agent_position = self.env.get_position()
        agent_direction = self.env.get_direction()
        state = agent_position + (agent_direction,)
        return state

    def render(self):
        plt.imshow(self.env.render())

# Testing
if __name__ == '__main__':
    from common.RandomKeyMEnv_10 import RandomEmptyEnv_10
    from common.KeyFlatObsWrapper import KeyFlatObsWrapper
    import matplotlib.pyplot as plt

    env = EnvWrapper(KeyFlatObsWrapper(RandomEmptyEnv_10(render_mode='rgb_array')))

    state = env.reset()
    print(f'start_state: {state}')