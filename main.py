import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../env_grid")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/agents")

import gym
import gym_grid
from agents.agent import Agent


class Flag_entity(object):
    hi = "hello"

class Flag(object):
    f = Flag_entity()

if __name__ == '__main__':

    f = Flag()
    print(f.hello)
    print(f.f.hi)
    exit()
    env = gym.make('grid-dsdp-v0')
    agent = Agent(env)
    agent.test()