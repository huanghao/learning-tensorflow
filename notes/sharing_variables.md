Sharing Variables
=================

https://www.tensorflow.org/programmers_guide/variable_scope

# The Problem

在一个函数里创建多个tf.Variable，在这个函数被重复调用的时候，这些变量会被重复再次创建。有一个办法就是把创建变量和使用变量的代码分离，但这种方式破坏了封装性。

tf提供了 Variable Scope的机制来解决这个问题

# Variable Scope Example

* tf.get_variable(<name>, <shape>, <initializer>): 创建或者返回一个命名变量
* tf.variable_scope(<scope_name>)

get_variable在不存在的时候去创建，使用initializer去初始化变量
- tf.constant_initializer(value)
- tf.random_uniform_initializer(a, b)
- tf.random_normal_initializer(mean, stddev)

  weights = tf.get_variable("weights", kernel_shape,
      initializer=tf.random_normal_initializer())

这个代码会创建名字叫weights的变量，但如果调用两次还是重复了，variable_scope就有用了

    with tf.variable_scope("conv1"):
      # Variables created here will be named "conv1/weights", "conv1/biases".
      relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])

tf.get_variable_scope().reuse
遇到相同的名字的时候，是报错还是重用
