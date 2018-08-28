# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

#### HYPER PARAMETERS ####
learning_rate = 0.0005
gamma = .99

n_hidden1 = 16
n_hidden2 = 16


class DQN:

    def __init__(self, state_dim, action_dim):
        tf.reset_default_graph()
        self.sess = tf.Session()

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.state_dim]) # one-hot representation
        self.action_ph = tf.placeholder(tf.int32, shape=[None])
        self.q_target_ph = tf.placeholder(tf.float32, shape=[None, 1])

        with tf.variable_scope("online"):
            self.online_network = self.generate_network(self.state_ph)  # Make online network
            
        with tf.variable_scope("target"):
            self.target_network = self.generate_network(self.state_ph)  # Make target network

        q_value = tf.reduce_sum(self.online_network * tf.one_hot(self.action_ph, action_dim),
                                axis=1, keepdims=True)
        error = tf.square(self.q_target_ph - q_value)
        loss = tf.reduce_mean(error)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_online_network = optimizer.minimize(loss)

        o_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        self.update_target_network = [tf.assign(t, o) for o, t in zip(o_params, t_params)]

        self.sess.run(tf.global_variables_initializer())

        return

    def generate_network(self, state_ph):

        hidden1 = tf.layers.dense(state_ph, n_hidden1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
        output = tf.layers.dense(hidden2, self.action_dim)

        return output

    def get_action(self, state):

        q_values = self.online_network.eval(session=self.sess, feed_dict={self.state_ph: [state]})
        return np.argmax(q_values)

    def train_network(self, state, action, reward, state_next, done):

        next_q_values = self.target_network.eval(session=self.sess, feed_dict={self.state_ph: state_next})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True).squeeze()
        q_target = (reward + (1.0 - done) * gamma * max_next_q_values).reshape((-1, 1))

        self.sess.run(self.train_online_network, 
                        feed_dict={self.state_ph: state, self.action_ph: action, self.q_target_ph: q_target})
            
        return

    def update_target(self):

        self.sess.run(self.update_target_network)

