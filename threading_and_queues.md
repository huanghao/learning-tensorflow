Threading and Queues
====================

https://www.tensorflow.org/programmers_guide/threading_and_queues

queue是graph上的一个节点，有状态的节点，想变量一样。其他的节点可以修改它的内容。其他的节点可以往queue里放数据或者拿数据。

# Queue usage overview

tf.FIFOQueue
tf.RandomShuffleQueue

典型的输入架构使用一个RandomShuffleQueue来准备训练需要的数据：

* 多线程准备数据并放到queue上
* 一个训练线程从queue上取mini-batches然后执行训练

tf.train.Coordinator: 用来帮助线程一起退出; 报告异常给等待它退出的程序
tf.train.QueueRunner: 用来创建一些线程往同一个queue上放数据

# Coordinator

* tf.train.Coordinator.should_stop: 返回真如果线程需要退出的话，用来等待退出条件的发生
* tf.train.Coordinator.request_stop: 线程发起退出请求，以后should_stop就返回True了
* tf.train.Coordinator.join: 等待所有的线程退出，主线程里使用

先创建一个Coordinator对象，然后创建一些线程。这些线程执行循环直到should_stop()返回True

任何一个线程要想使整个计算结束时，必须要调用request_stop()，然后其他线程会在should_stop()返回True的时候也退出。

https://www.tensorflow.org/api_docs/python/tf/train/Coordinator

## join()

request stop with an exception

一个线程在调用request_stop的时候传递了一个异常，这个异常会在join函数上被抛出(reraise)
但是stacktrace有问题，显示的是 `six.reraise(*self._exc_info_to_raise)` 这个是tf的一段代码实现，而不是真正发生异常的thread本身的栈。看coord_request_stop_with_exec.py

这个异常必须是在try/except的上下文中，而不是一个新创建的异常

with coord.stop_on_exception() 用来简化thread抛出异常的代码

Grace period for stopping

当一个线程request_stop之后，默认别的线程有2分钟时间来退出，这个时间叫做优雅的退出时间。但如果超过这个时间，还有线程没有退出，那么join会抛出一个RuntimeException

## __init__()

clean_stop_exception_types=(tf.errors.OutOfRangeError,)
构造的时候可以指定一个异常的列表，request_stop给定的异常在这个范围之内的可以被忽略，相当于正常的停止。
默认的OutOfRangeError表示queue被消费完了，类似于StopIteration

# QueueRunner

这个类创建一批线程重复执行进队列操作。这些线程可以使用coordinator来一起结束。另外，一个queue runner会启动一个"关闭线程"来自动的关闭queue，如果一个异常被报告到coordinator上的时候。

qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
qr.create_threads(sess, coord=coord, start=True)

这句描述解释得非常清楚

  Holds a list of enqueue operations for a queue, each to be run in a thread.

QueueRunner接受两个东西，1是queue本身，2是一些enqueue操作。最后每个操作都会被运行在一个单独的线程里



# Handling exceptions

那些被queue runners启动的线程除了做进队列操作以外，还做一些别的事情。他们可以捕获和处理queue产生的异常，包括tf.errors.OutOfRangeError（表示queue已经被关闭了）

所以一个使用了coordinator的程序也必须要处理这些异常。

  try:
    for step in xrange(10000):
      if coord.should_stop():
        break
      sess.run(train_op)
  except Exception as e:
    # Report exceptions to the coordinator
    coord.request_stop(e)
  finally:
    # Terminate as usual. It is safe to call `coord.request_stop()` twice.
    coord.request_stop()
    coord.join(threads)

遇到异常的时候或者正常退出的时候都需要调用request_stop。
最后都需要调用join等待所有线程退出。
