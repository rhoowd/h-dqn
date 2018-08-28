from __future__ import print_function

import tensorflow as tf
import logging
import time

is_testing = True

if is_testing:
    class Flags_entity(object):
        env = 'grid-dsdp-v0'
        agent = "qlearn"
        train_step = 1000
        test_step = 1000
        test_period = 200
        max_step_per_episode = 1000

    class Flags(object):
        FLAGS = Flags_entity()

    flags = Flags()

else:
    flags = tf.flags
    flags.DEFINE_string("env", "grid-dsdp-v0", "Environment")

    # BaseAgent Setting
    flags.DEFINE_integer("train_step", 200, "Training step")
    flags.DEFINE_integer("test_step", 100, "Testing step")
    flags.DEFINE_integer("test_period", 50, "Testing period")
    flags.DEFINE_integer("max_step_per_episode", 1000, "max_step_per_episode")

    # Agent setting
    flags.DEFINE_string("agent", "qlearn", "Name of agent")


now = time.localtime()
s_time = "%02d%02d%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
filename = flags.FLAGS.agent + "-" + s_time


result = logging.getLogger('Result')
result.setLevel(logging.INFO)
result_fh = logging.FileHandler("./agents/results/eval/r-" + filename + ".txt")
result_fm = logging.Formatter('[%(filename)s:%(lineno)s] %(asctime)s\t%(message)s')
result_fh.setFormatter(result_fm)
result.addHandler(result_fh)

result_sh = logging.StreamHandler()
result_sh.setFormatter(result_fm)
result.addHandler(result_sh)