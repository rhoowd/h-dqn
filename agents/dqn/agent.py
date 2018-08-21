# coding=utf8
from __future__ import print_function
from __future__ import division
import numpy as np

from agents.agent import Agent as BaseAgent
from agents.dqn.dqn_model import DQN
from agents.common.replay_buffer import ReplayBuffer

minibatch_size = 32
pre_train_step = 10
target_update_period = 1000

class Agent(BaseAgent):

    def __init__(self, env):
        super(Agent, self).__init__(env)

        self.model = DQN(self.obs_dim, self.action_dim)
        self.replay_buffer = ReplayBuffer(minibatch_size=minibatch_size)

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

        self.replay_buffer.add_to_memory((obs, action, reward, obs_next, done))
        
        if len(self.replay_buffer.replay_memory) < minibatch_size * pre_train_step:
            return None

        minibatch = self.replay_buffer.sample_from_memory()
        s, a, r, s_, done = map(np.array, zip(*minibatch))
        self.model.train_network(s, a, r, s_, done)

        if self.global_step % target_update_period == 0:
            self.model.update_target()

        return
