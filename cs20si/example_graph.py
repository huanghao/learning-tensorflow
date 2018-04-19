"""
➜  cs20si git:(master) ✗ python example_graph.py | python test_node_def.py | dot -Tpng > example_graph.png; open example_graph.png
"""
import sys
import tensorflow as tf

x = tf.Variable(2., name='x')
y = tf.Variable(3., name='y')
z = tf.exp(x)

grad_z = tf.gradients(z, [x, y])

print(tf.get_default_graph().as_graph_def())
sys.exit(0)

with tf.Session() as s:
    s.run([x.initializer, y.initializer])
    print(s.run([z, grad_z]))
