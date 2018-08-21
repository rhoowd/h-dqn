# coding=utf8
from __future__ import print_function
from __future__ import division
import numpy as np

from agents.agent import Agent as BaseAgent


class Agent(BaseAgent):

    def __init__(self, env):
        super(Agent, self).__init__(env)

        # hyper parameter setting 
        self.df = .99   # discount factor
        self.lr = 0.01   # learning rate

        # Make Q-table
        self.q_table = np.zeros([self.obs_dim, self.action_dim])

        self.set_gui_flag(False, False)

    def get_action(self, obs, train=True):

        epsilon = 1. / ((self.global_step // 10) + 1)
        if train and np.random.rand(1) < epsilon:  # Exploration
            action = self.env.action_space.sample()
        else:  # Select best action
            action =int(np.argmax(self.q_table[obs, :]))
        return action

    def train_model(self, obs, action, reward, obs_next, done):

        self.q_table[obs, action] = \
            (1-self.lr) * self.q_table[obs, action] \
            + self.lr * (reward + self.df * np.max(self.q_table[obs_next,:]))
        return None
