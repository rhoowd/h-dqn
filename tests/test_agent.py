import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../agents")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../env_grid")

from agents.agent import Agent
import gym
import gym_grid


@pytest.fixture
def agent():
    env = gym.make('grid-dsdp-v0')
    agent = Agent(env)
    yield agent


def test_create_agent(agent):
    assert True

def test_get_action_format_check(agent):  
    obs = agent.env.reset()
    action = agent.get_action(obs)
    agent.env.action_space.contains(action)

@pytest.mark.parametrize("step",[1,5,100])
def test_learn_steps_and_episode_num(agent, step):  
    agent.train_step = step
    max_step = agent.max_step_per_episode
    agent.learn()
    assert agent.global_step == step


def test_learn_multiple_times(agent):
    step = agent.train_step
    max_step = agent.max_step_per_episode
    agent.learn()
    agent.learn()
    assert agent.global_step == 2 * step

    
def test_train_model(agent):
    obs = agent.env.reset()
    action = agent.get_action(obs)
    obs_next, reward, done, info = agent.env.step(action)
    result = agent.train_model(obs, action, reward, obs_next, done)
    assert result


@pytest.mark.parametrize("step",[1,5,100])
def test_test_steps_and_episode_num(agent, step):  
    agent.test_step = step
    agent.test()
    assert agent.t_step == step


def test_tensorboard(agent):
    agent.learn()
    

    

