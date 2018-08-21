from __future__ import print_function


class Agent(object):
    def __init__(self, env):
        self.env = env
        self.train_step = 100
        self.test_step = 100
        self.max_step_per_episode = 15

        self.action_dim = env.action_space.n
        self.obs_dim = env.observation_space.n
        
        self.global_step = 0
        self.l_step = 0
        self.l_episode_step = 0
        self.episode_num = 0
        self.t_step = 0
        self.t_episode_step = 0

        self.gui_flag_l = False
        self.gui_flag_t = True

    def learn(self):
        self.l_step = 0

        while not self.is_learn_done():
            self.reset_episode_in_learn()
            obs = self.env.reset()
            done = False
            if self.gui_flag_l:
                self.env.render()

            while not (self.is_episode_done(done) or self.is_learn_done()):
                self.increase_step_in_learn()

                action = self.get_action(obs)
                obs_next, reward, done, _ = self.env.step(action)
                self.train_model(obs, action, reward, obs_next, done)
                obs = obs_next

                if self.gui_flag_l:
                    self.env.render()

    def test(self):
        self.t_step = 0
        while not self.is_learn_done_in_test():
            self.reset_episode_in_test()
            obs = self.env.reset()
            done = False
            if self.gui_flag_t:
                self.env.render()
            
            while not (self.is_episode_done_in_test(done) or self.is_learn_done_in_test()):
                self.increase_step_in_test()                

                action = self.get_action(obs, train=False)
                obs_next, reward, done, _ = self.env.step(action)
                obs = obs_next

                if self.gui_flag_t:
                    self.env.render()


    def get_action(self, obs, train=True):
        print("get_action is not implemented !")
        return self.env.action_space.sample()

    def train_model(self, obs, action, reward, obs_next, done):
        print("train_model is not implemented !")
        return True

    def is_episode_done(self, done):
        return self.l_episode_step >= self.max_step_per_episode or done

    def is_learn_done(self):
        return self.l_step >= self.train_step

    def is_episode_done_in_test(self, done):
        return self.t_episode_step >= self.max_step_per_episode or done

    def is_learn_done_in_test(self):
        return self.t_step >= self.test_step

    def increase_step_in_learn(self):
        self.global_step += 1
        self.l_step += 1
        self.l_episode_step += 1

    def reset_episode_in_learn(self):
        self.episode_num += 1
        self.l_episode_step = 0

    def increase_step_in_test(self):
        self.t_episode_step += 1
        self.t_step += 1       

    def reset_episode_in_test(self):
        self.t_episode_step = 0

    def set_gui_flag(self, learn, test=True):
        self.gui_flag_l = learn
        self.gui_flag_t = test