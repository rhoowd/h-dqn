# coding=utf8
from __future__ import print_function
from __future__ import division
import numpy as np

from agents.agent import Agent as BaseAgent
from agents.hdqn.hdqn_model import hDQN
from agents.common.replay_buffer import ReplayBuffer

class Agent(BaseAgent):

    def __init__(self, env):
        super(Agent, self).__init__(env)
        self.model = hDQN(self.obs_dim, self.action_dim)

        self.set_gui_flag(False, False)

    def get_action(self, obs, train=True):
        eps_min = 0.1
        eps_max = 1.0
        eps_decay_steps = self.train_step
        epsilon = max(eps_min, eps_max - (eps_max - eps_min)*self.global_step/eps_decay_steps)

        if train and np.random.rand(1) < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.model.get_action(obs)

        return action


    def train_model(self, obs, action, reward, obs_next, done):
        print("[hDQN] train_model is not implemented !")
        return True

    def reset_episode(self):
        self.model.reset_episode()