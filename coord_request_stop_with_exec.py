import threading
import tensorflow as tf

def child(coord):
    try:
        1/0
    except Exception as e:
        coord.request_stop(e)

def child2(coord):
    with coord.stop_on_exception():
        1/0

def child3(coord):
    coord.request_stop(Exception('xxx'))  # wrong

coord = tf.train.Coordinator()
t = threading.Thread(target=child2, args=(coord,))
t.run()

try:
    coord.join([t])
except Exception as e:
    print 'here:', e
    import traceback
    traceback.print_exc()

