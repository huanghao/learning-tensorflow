Reading data
============

https://www.tensorflow.org/programmers_guide/reading_data

三种主要的提供数据的方法：

- Feeding，喂
- 读文件
- Reload data，把数据直接提前load到tf.constant或者tf.variable里

# Feeding

run()/eval() 里的参数，接收一个字典，feed_dict={...}
字典的key是tensor变量（注意不是字符串）
这个方法可以替换任何的tensor，但推荐用tf.placeholder节点。placeholder就是为了这个目的而存在的。

# Reading from files

一个典型的数据管道包含几个步骤：

1. 文件名列表
2. 打乱文件名顺序（可选）
3. epoch限制（可选）
4. 文件名队列
5. 文件格式对应的Reader
6. 对于文件里的记录的解码器decoder
7. 预处理（可选）
8. Example队列

## 文件名，洗牌和epoch限制

文件名列表，可以是constant string tensor，类似于["file0", "file1"]这样，或者是
tf.train.match_filenames_once()函数

把文件名列表给tf.train.string_input_producer()来创建一个FIFO的队列，包含这些文件名，只到reader读走
