import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class EnergyEnv(gym.Env):
    def __init__(self,training_data):
        # data
        self.data_size_history = training_data
        self.n_step = self.data_size_history.size

        # instance attributes
        self.cur_step = None
        self.data_size = None
        self.prev = 0
        # calculation variables
        self.energy_4G = 100;
        self.energy_wifi = 10;
        self.bandwidth_4G = 15;
        self.bandwidth_wifi = 150;
        self.beta = 10;

        # action space
        self.action_space = spaces.Discrete(1)


        # observation space
        self.observation_space = spaces.Discrete(1)

        self._seed()
        self._reset()

    def _seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _reset(self):
        self.cur_step = 0
        self.data_size = int(self.data_size_history[self.cur_step])
        return self._get_obs(0)

    def _step(self,action):
        #assert self.action_space.contains(action)

        self.cur_step += 1
        self.data_size = self.data_size_history[self.cur_step]
        cur =  self._compute(self.data_size,action)
        if(cur < self.prev):
            reward = 1
        elif(cur > self.prev):
            reward = -1
        else:
            reward = 0
        self.prev = cur;
        done = self.cur_step == self.n_step -1
        info = {'cur_val': reward}

        return self._get_obs(1), reward , done, info

    def _compute(self,data_size,action):
        data_size = int(data_size)
        action = int(action)
        energy = (self.energy_wifi*data_size*action)+(self.energy_4G*data_size*(1-action))
        delay1 = self.bandwidth_4G*data_size*(1-action)
        delay2 = self.bandwidth_wifi*data_size
        m_delay = self.beta*max(delay1,delay2)
        return energy + m_delay

    def _get_obs(self,tag):
        obs_arry = np.array([self.data_size])

        return obs_arry














