import tensorflow as tf
import config

class TensorBoard(object):

    def __init__(self):
        self.sess = tf.Session()
        self.file_name = "./agents/results/eval/"+config.filename+"/"
        self.graphs = dict()
        self.labels = dict()

    def add_graph(self, graph_name):
        if graph_name in self.graphs:
            assert False  # Duplicated graph name

        var = tf.Variable(0.)
        op =tf.summary.scalar(graph_name, var)

        self.graphs[graph_name] = dict()
        self.graphs[graph_name]['var'] = var
        self.graphs[graph_name]['op'] = op
        self.labels[graph_name] = dict()

    def add_label(self, graph_name, label_name):
        if label_name in self.labels[graph_name]:
            assert False  # Duplicated label name
        writer = tf.summary.FileWriter(self.file_name +graph_name+"/"+label_name, self.sess.graph)
        r = self.sess.run(self.graphs[graph_name]['op'], feed_dict={self.graphs[graph_name]['var']: 0})
        writer.add_summary(r, 0)
        self.labels[graph_name][label_name] = writer

    def write(self, graph_name, label_name, value, step):
        r = self.sess.run(self.graphs[graph_name]['op'], feed_dict={self.graphs[graph_name]['var']: value})
        self.labels[graph_name][label_name].add_summary(r, step)
