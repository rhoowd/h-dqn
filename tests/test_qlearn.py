import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../agents")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../env_grid")

from agents.qlearn.agent import Agent
import gym
import gym_grid


@pytest.fixture
def agent():
    env = gym.make('grid-dsdp-v0')
    agent = Agent(env)
    yield agent

def test_get_action_format_check(agent):  
    obs = agent.env.reset()
    action = agent.get_action(obs)
    agent.env.action_space.contains(action)

def test_run_learn(agent):
    agent.learn()
