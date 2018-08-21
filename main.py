import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../env_grid")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/agents")

import gym
import gym_grid
from agents.agent import Agent

print("hello")

if __name__ == '__main__':

    env = gym.make('grid-dsdp-v0')
    agent = Agent(env)
    agent.test()