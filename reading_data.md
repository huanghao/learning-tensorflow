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
  constant string tensor
  tf.train.match_filenames_once()
2. 打乱文件名顺序（可选）
  string_input_producer(shuffle=True)
3. epoch限制（可选）
  string_input_producer(num_epochs=None)
4. 文件名队列
  string_input_producer
5. 文件格式对应的Reader
  tf.TextLineReader -> tf.decode_csv
  tf.FixedLengthRecordReader -> tf.decode_raw
  tf.TFRecordReader -> tf.parse_single_example
6. 对于文件里的记录的解码器decoder
7. 预处理（可选）
8. Example队列

## 文件名，洗牌和epoch限制

文件名列表，可以是constant string tensor，类似于["file0", "file1"]这样，或者是
tf.train.match_filenames_once()函数
https://www.tensorflow.org/api_docs/python/tf/train/match_filenames_once

把文件名列表给tf.train.string_input_producer()来创建一个FIFO的队列，包含这些文件名，只到reader读走
https://www.tensorflow.org/api_docs/python/tf/train/string_input_producer

string_input_producer可以控制位置变换和设置最大的epoch次数，queue runner在一个epoch里把整个文件名列表加入到queue里一次，如果shuffle=True就会打乱文件名的顺序，这个过程生成的是文件名的均匀分布，所以每个样本都不会比另外一个更多或者更少的被采样。

queue runner和reader工作在不同的线程里，所以打乱顺序或者往队列里增加文件名都不会阻塞reader

## File formats

选择一个匹配格式的reader，把queue给reader的read()方法。
这个read方法就会返回一个key和一个value。
key用来标识这个文件或者记录（在debugging的时候有用）
value是一个字符串标量，再使用一个或者多个decoder或者conversion ops去把这个字符串解析成tensors

### CSV files

读csv文件用这两类，tf.TextLineReader 按行读取，tf.decode_csv 把csv解析成tensor的列表

在调用run/eval之前，必须先调用tf.train.start_queue_runners，不然read操作会被阻塞在队列上等待文件名
tf.train.Coordinator() 是什么？

### Fixed length record

tf.FixedLengthRecordReader读取定长记录，tf.decode_raw把字符串转成uint8类型的tensor
https://www.tensorflow.org/code/tensorflow_models/tutorials/image/cifar10/cifar10_input.py
这个例子是读取CIFAR-10的一个例子，看一下

### Standard TensorFlow format

推荐使用TFRecord files格式来统一各种格式。
https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details

这种格式包含了tf.train.Example，它由包含了很多Features作为列。
https://www.tensorflow.org/code/tensorflow/core/example/example.proto
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto

  message Example {
    Features features = 1;
  };

  message Features {
    // Map from feature name to feature.
    map<string, Feature> feature = 1;
  };

  // Containers for non-sequential data.
  message Feature {
    // Each feature can be exactly one kind.
    oneof kind {
      BytesList bytes_list = 1;
      FloatList float_list = 2;
      Int64List int64_list = 3;
    }
  };

可以看到Example就是一些Features，而Features就是string到Feature的一个map，一个Feature是bytes/float/int64 list三者之一

写一个程序把数据加载了以后填充到一个Example结构中，然后通过protocol buffer序列化成一个字符串，通过tf.python_io.TFRecordWriter 把这个字符串写成TFRecord file

https://www.tensorflow.org/code/tensorflow/examples/how_tos/reading_data/convert_to_records.py
这个例子把MNIST转成了这种格式，可以看一下

读取TFRecords文件的时候，使用tf.TFRecordReader和tf.parse_single_example 会把example protocol buffer解码成tensor

## Preprocessing

可以对上面拿到的Example做任何需要的预处理操作。例如标准化，随机选取，加噪音或者变形扭曲。

https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py#L140
这个例子里的distorted_inputs函数有几种操作：

- tf.random_crop 随机剪切图片中的一个大小[height, width]的区域
- tf.image.random_flip_left_right 随机水平翻转一个图片
- tf.image.random_brightness 随机亮度
- tf.image.random_contrast 随机对比度
- tf.image.per_image_standardization 根据mean/variance做标准化

## Batching

在队列的最后我们使用另一个队列把样本成组，以便做train/evaluation/inference
