from __future__ import print_function

import tensorflow as tf
import logging
import time

is_testing = True

if is_testing:
    class Flags_entity(object):
        agent = "dqn"
        train_step = 100000
        test_step = 1000
        test_period = 200

    class Flags(object):
        FLAGS = Flags_entity()

    flags = Flags()

else:
    flags = tf.flags
    flags.DEFINE_string("agent", "qlearn", "Name of agent")
    flags.DEFINE_integer("train_step", 200, "Training step")
    flags.DEFINE_integer("test_step", 100, "Testing step")
    flags.DEFINE_integer("test_period", 50, "Testing period")


now = time.localtime()
s_time = "%02d%02d%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
filename = flags.FLAGS.agent + "-" + s_time
