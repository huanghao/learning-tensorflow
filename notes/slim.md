https://github.com/tensorflow/models/tree/master/research/slim
----

slim是轻量级的，高层api，用来训练和评估复杂的模型。包含了用来训练评估cnn图片分类的模型。
从零开始训练，或者fine-tune。也包含下载标准数据集，并转成tfrecord格式，并使用queue相关的工具。

数据集
Flowers， Cifar10，MNIST，ImageNet
download_and_convert_data.py 会下载并把它们转换成tfrecord格式

https://github.com/tensorflow/models/blob/master/research/inception/README.md#getting-started
这个地方介绍了，怎么使用imagenet从头开始训练inception v3
值得一看

pre-trained models


https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb
----

