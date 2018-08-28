import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../agents")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../env_grid")

from agents.hdqn.agent import Agent
import gym
import gym_grid


@pytest.fixture
def agent():
    env = gym.make('grid-dsdp-v1')
    agent = Agent(env)
    yield agent


def test_create_agent(agent):
    assert True


def test_get_action(agent):
    obs = agent.env.reset()
    action = agent.model.get_action(obs)
    assert agent.env.action_space.contains(action)
    action = agent.model.get_action(obs)
    assert agent.env.action_space.contains(action)



def test_model_generate_goal_nn(agent):
    g_nn = agent.model.generate_goal_network(agent.model.state_ph)    
    assert g_nn.shape[1] == agent.model.state_dim


def test_model_generate_action_nn(agent):
    a_nn = agent.model.generate_action_network(agent.model.state_ph, agent.model.goal_ph)
    print a_nn
    assert True

def test_get_action_for_goal(agent):
    obs = agent.env.reset()
    goal = agent.env.one_hot(0)
    action = agent.model.get_action_for_goal(obs, goal)
    assert agent.env.action_space.contains(action)

def test_get_goal(agent):
    obs = agent.env.reset()
    goal = agent.model.get_goal(obs)
    assert agent.env.observation_space.contains(goal)


def test_goal_mgr(agent):
    goal = agent.env.reset()
    agent.model.goal_mgr.set_goal(goal)
    assert np.array_equal(agent.model.goal_mgr.get_goal(), goal)
    assert agent.model.goal_mgr.is_goal_achieved(goal)
    wrong_goal = agent.env.one_hot(0)
    assert not agent.model.goal_mgr.is_goal_achieved(wrong_goal)

def test_goal_mgr_reset(agent):
    agent.reset_episode()
    assert not agent.model.goal_mgr.is_initiated()
    agent.model.goal_mgr.set_goal(agent.env.one_hot(0))
    assert agent.model.goal_mgr.is_initiated()



def test_run_learn(agent):
    agent.learn()