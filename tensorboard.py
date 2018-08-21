import tensorflow as tf

def make_summary():
    v_list = list()
    v_list.append(tf.Variable(0.))
    tf.summary.scalar('a', v_list[0])
    op = tf.summary.merge_all()

    return v_list, op

with tf.Session() as sess:
    writer = tf.summary.FileWriter('.', sess.graph)

    v_list, op = make_summary()

    r = sess.run(op, feed_dict={v_list[0]: 3})
    writer.add_summary(r, 0)

    r = sess.run(op, feed_dict={v_list[0]: 5})
    writer.add_summary(r, 1)