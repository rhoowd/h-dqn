import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../env_grid")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/agents")

import gym
import gym_grid
from agents.agent import Agent
import agents
import agents.config

FLAGS = agents.config.flags.FLAGS

if __name__ == '__main__':
    # set_seed(FLAGS.seed)

    # Load environment
    print('Environment: {}'.format(FLAGS.env))
    env = gym.make(FLAGS.env)

    # Load agent
    print('Agent: {}'.format(FLAGS.agent))
    agent = agents.load(FLAGS.agent+"/agent.py").Agent(env)

    # start learning
    agent.learn()


