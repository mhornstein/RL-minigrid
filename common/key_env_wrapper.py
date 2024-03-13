import gym
import copy
import matplotlib.pyplot as plt
from collections import namedtuple

agent_directions_space = 4
key_state_space = 2
door_state_space = 2

GoalReward = namedtuple('GoalReward', ['key', 'door'])

class KeyEnvWrapper(gym.Env):
    '''
    This wrapper enables environment customization
    '''
    def __init__(self, env, step_reward=-0.01, goal_reward=GoalReward(key=1, door=5)):
        '''
        Initializes the EnvWrapper.

        Args:
            env (gym.Env): The Gym environment to be wrapped. Important: this env must be initialized before wrapping (use env.reset())
            step_reward (float, optional): The per-step reward or penalty. Default is -0.01.
            goal_reward (GoalReward, optional): A namedtuple `GoalReward(key, door)` specifying the rewards for the specific events within the environment.
                                                 - `key` (float): The reward for the event of picking up the key.
                                                 - `door` (float): The reward for the event of opening the door.
                                                 Default is GoalReward(key=1, door=5).
        '''
        self.source_env = env
        self.reset()

        self.set_params(step_reward, goal_reward)

    def reset(self): # open gym default implementation changes the board. override, as we do not want to change the board upon reset every time
        self.env = copy.deepcopy(self.source_env)
        return self.get_current_state()

    def set_params(self, step_reward, goal_reward):
        self.step_reward = step_reward
        self.goal_reward = goal_reward

    def get_current_state(self):
        '''
        The state holds the knowledge of the agent: the agent knows where it is and what its direction is,
        as well as some history of its actions: whether it has picked up the key or unlocked the door
        '''
        agent_col, agent_row = self.env.get_position()
        agent_direction = self.env.get_direction()
        is_carrying_key = self.env.is_carrying_key()
        is_door_opened = self.env.is_door_open()
        state = (agent_col - 1, agent_row - 1, agent_direction, int(is_carrying_key), int(is_door_opened))
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
        return self.env.action_space.sample()

    def step(self, action):
        # First - log current state
        agent_had_key = self.env.is_carrying_key()
        door_was_opened = self.env.is_door_open()

        # now - perform the action and check its outcome
        _ = self.env.step(action)
        s_tag = self.get_current_state()
        agent_has_key = self.env.is_carrying_key()
        door_is_opened = self.env.is_door_open()

        done = not door_was_opened and door_is_opened # the door was just opened

        r = self.step_reward
        if done: # the agent unlocked the door
            r = self.goal_reward.door
        elif not agent_had_key and agent_has_key: # the agent picked up the key
            r = self.goal_reward.key

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