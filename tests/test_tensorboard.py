import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../agents")

# tensorboard --logdir=.

from agents.tensorboard import TensorBoard

@pytest.fixture
def tb():
    tb = TensorBoard()
    yield tb


def test_create_class(tb):
    print("hi")

def test_draw_one_label_on_one_graph(tb):
    tb.add_graph('graph1')
    tb.add_label('graph1','label1')
    value = 1
    step = 1
    tb.write('graph1','label1', value, step)

def test_draw_one_label_on_two_graph(tb):
    tb.add_graph('graph1')
    tb.add_graph('graph2')
    tb.add_label('graph1','label1')
    tb.add_label('graph2','label2')
    tb.write('graph1','label1', 1, 1)
    tb.write('graph1','label1', 1, 2)    
    tb.write('graph1','label1', 1, 3)    
    tb.write('graph1','label1', 1, 4)    
    tb.write('graph2','label2', 2, 1)
    tb.write('graph2','label2', 2, 2)
    tb.write('graph2','label2', 2, 3)
    tb.write('graph2','label2', 2, 4)


def test_draw_two_label_on_two_graph(tb):
    tb.add_graph('graph1')
    tb.add_graph('graph2')
    tb.add_label('graph1','label1')
    tb.add_label('graph1','label2')
    tb.add_label('graph2','label2')
    tb.add_label('graph2','label4')
    for i in range (10):
        tb.write('graph1','label1', 1, i)
        tb.write('graph1','label2', 3, i)
        tb.write('graph2','label2', 2, i)
        tb.write('graph2','label4', 5, i)
