https://www.tensorflow.org/performance/
====

https://www.tensorflow.org/performance/performance_guide
----

input pipeline: 从磁盘读文件，decode成tensor，做数据预处理。这个过程被称为预处理
CPU-to-GPU Data Transfer: Transfer images from CPU to GPU.

Some cloud solutions have network attached disks that start as low as 50 MB/sec, which is slower than spinning disks (150 MB/sec), SATA SSDs (500 MB/sec), and PCIe SSDs (2,000+ MB/sec).

Placing input pipeline operations on the CPU can significantly improve performance.

If using tf.estimator.Estimator the input function is automatically placed on the CPU.

The tf.data API utilizes C++ multi-threading and has a much lower overhead than the Python-based queue_runner that is limited by Python's multi-threading performance.

Our recommendation is to avoid using feed_dict for all but trivial examples. In particular, avoid using feed_dict with large inputs:

If inputs are JPEG images that also require cropping, use fused tf.image.decode_and_crop_jpeg to speed up preprocessing.
For imagenet data, this approach could speed up the input pipeline by up to 30%.

Q: 测量手段是什么？怎么得到这个数据，如果我们要自己做优化的情况下

One approach to get maximum I/O throughput is to preprocess input data into larger (~100MB) TFRecord files. For smaller data sets (200MB-1GB), the best approach is often to load the entire data set into memory.

NHWC is the TensorFlow default and NCHW is the optimal format to use when training on NVIDIA GPUs using cuDNN.

Q: batch设多少怎么能有量化的方法，而不是像现在一样去猜？

https://www.tensorflow.org/performance/performance_models
----

Most TensorFlow operations used by a CNN support both NHWC and NCHW data format. On GPU, NCHW is faster. But on CPU, NHWC is sometimes faster.
Build the model with both NHWC and NCHW
Q: 怎么实现？

https://www.tensorflow.org/performance/datasets_performance
----

ETL: E: 从文件系统读数据，T: 解码做变换，数据增强之类，随机，batch，L: cpu到gpu

https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks
----

https://www.tensorflow.org/performance/benchmarks#methodology
TODO: 这里提到的这个工具，可以了解一下，可能是个标准的工具，用来对不同的硬件或者不同的模型跑benchmark能出个报告

https://www.tensorflow.org/performance/quantization
----

TensorFlow Lite adds quantization that uses an 8-bit fixed point representation.

Since fetching 8-bit values only requires 25% of the memory bandwidth of floats, more efficient caches avoid bottlenecks for RAM access.

Typically, SIMD operations are available that run more operations per clock cycle. In some cases, a DSP chip is available that accelerates 8-bit calculations resulting in a massive speedup.

https://www.tensorflow.org/performance/xla/
----

暂时还在开发阶段，不一定能看到提升，但可以了解它的架构和大概的实现原理，以后跟进
