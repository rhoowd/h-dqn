# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

#### HYPER PARAMETERS ####
learning_rate = 0.0005
gamma = .99

n_hidden_goal_1 = 16
n_hidden_goal_2 = 16

n_hidden_action_1 = 16
n_hidden_action_2 = 16

class GoalManager(object):
    
    def __init__(self):
        self.goal = None
        self.init_flag = False

    def reset(self):
        self.goal = None
        self.init_flag = False

    def set_goal(self, goal):
        self.init_flag = True
        self.goal = goal

    def get_goal(self):
        return self.goal

    def is_goal_achieved(self, obs):
        return np.array_equal(self.goal, obs)

    def is_initiated(self):
        return self.init_flag
        

class hDQN:

    def __init__(self, state_dim, action_dim):
        tf.reset_default_graph()
        self.sess = tf.Session()

        self.goal_mgr = GoalManager()

        self.state_dim = state_dim
        self.goal_dim = state_dim
        self.action_dim = action_dim

        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.state_dim]) # one-hot representation
        self.goal_ph = tf.placeholder(tf.float32, shape=[None, self.state_dim]) # one-hot representation



        with tf.variable_scope("online"):
            self.online_goal_network = self.generate_goal_network(self.state_ph)  # Make online network
            self.online_action_network = self.generate_action_network(self.state_ph, self.goal_ph)  # Make online network
            
        with tf.variable_scope("target"):
            self.target_goal_network = self.generate_goal_network(self.state_ph)  # Make target network
            self.target_action_network = self.generate_action_network(self.state_ph, self.goal_ph)  # Make online network

        self.sess.run(tf.global_variables_initializer())


    def generate_goal_network(self, state_ph):

        hidden1 = tf.layers.dense(state_ph, n_hidden_goal_1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden_goal_2, activation=tf.nn.relu)
        output = tf.layers.dense(hidden2, self.goal_dim)

        return output

    def generate_action_network(self, state_ph, goal_ph):
        obs_goal_input = tf.concat([state_ph, goal_ph], axis=-1)

        hidden1 = tf.layers.dense(obs_goal_input, n_hidden_action_1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden_action_2, activation=tf.nn.relu)
        output = tf.layers.dense(hidden2, self.action_dim)

        return output


    def get_action_for_goal(self, state, goal):
        
        q_values = self.online_action_network.eval(session=self.sess, feed_dict={self.state_ph: [state], self.goal_ph: [goal]})
        return np.argmax(q_values)

    def get_new_goal(self, state):

        q_values = self.online_goal_network.eval(session=self.sess, feed_dict={self.state_ph: [state]})
        i = np.argmax(q_values)
        ret = np.zeros(self.goal_dim)
    	ret[i] = 1
        return ret

    def get_action(self, state):
        if self.goal_mgr.is_goal_achieved(state) or not self.goal_mgr.is_initiated():
            new_goal = self.get_new_goal(state)
            self.goal_mgr.set_goal(new_goal)
        
        goal = self.goal_mgr.get_goal()
        action = self.get_action_for_goal(state, goal)  
        return action

    def reset_episode(self):
        self.goal_mgr.reset()
