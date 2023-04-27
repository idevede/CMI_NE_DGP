import gym
from gym import spaces

class ChemPOMDPWrapper(gym.Wrapper):
    def __init__(self, env, hidden_objects_ind: list, hidden_targets_ind: list):
        super().__init__(env)
        full_obs_dims = range(len(env.observation_dims()))
        hidden_dims = hidden_objects_ind + [env.num_objects + i for i in hidden_targets_ind]
        self.partial_obs_dims = [i for i in full_obs_dims if i not in hidden_dims]

        assert 0 < len(self.partial_obs_dims) <= self.observation_space.shape[0]

        self.action_space = spaces.Discrete(self.num_actions-len(hidden_objects_ind))

        # self.observation_space = spaces.Box(
        #     low=self.observation_space.low[self.partial_obs_dims],
        #     high=self.observation_space.high[self.partial_obs_dims],
        #     dtype=int,
        # )

    def observation_dims(self):
        state = self.env.observation_dims()
        partial_obs_keys = [list(state.keys())[i] for i in self.partial_obs_dims]
        partial_state = {key: state[key] for key in partial_obs_keys}
        return partial_state


    # def get_obs(self, state):
    #     return state[self.partial_obs_dims].copy()

    def get_state(self):
        state = self.env.get_state()
        partial_obs_keys = [list(state.keys())[i] for i in self.partial_obs_dims]
        partial_state = {key: state[key] for key in partial_obs_keys}
        return partial_state



    # def reset(self):
    #     state, info = self.env.reset()
    #     return self.get_obs(state), info
    #
    # def step(self, action):
    #     state, reward, terminated, truncated, info = self.env.step(action)
    #     return self.get_obs(state), reward, terminated, truncated, info


# if __name__ == "__main__":
#     import envs
#
#     env = gym.make("ColorChangingRLPOMDP-3-3-Static-10-v0")
#     obs = env.reset()
#     done = False
#     step = 0
#     while not done:
#         next_obs, rew, done, info = env.step(env.action_space.sample())
#         step += 1
#         print(step, done, info)
