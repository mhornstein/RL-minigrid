import gym
import copy
import matplotlib.pyplot as plt

agent_directions_space = 4

class EnvWrapper(gym.Env):
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

        self.step_reward = step_reward
        self.goal_reward = goal_reward

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

    def step(self, action):
        _ = self.env.step(action)
        s_tag = self.get_current_state()
        done = self.env.get_goal_pos() == self.env.get_position() # compare agent location to target location
        r = self.goal_reward if done else self.step_reward
        return s_tag, r, done

# Testing
if __name__ == '__main__':
    from common.RandomKeyMEnv_10 import RandomEmptyEnv_10
    from common.KeyFlatObsWrapper import KeyFlatObsWrapper
    import matplotlib.pyplot as plt

    source_env = KeyFlatObsWrapper(RandomEmptyEnv_10(render_mode='rgb_array'))
    source_env.reset()  # we must reset the env: this places the agent and other elements on top of the board
    env = EnvWrapper(source_env)

    state = env.reset()
    print(f'start_state: {state}')

    print(f'states dim: {env.get_state_dim()}')
    print(f'action dim: {env.get_action_dim()}')

    action = env.sample_action()
    print(f'sampled action: {action}')
    s_tag, r, done = env.step(action)