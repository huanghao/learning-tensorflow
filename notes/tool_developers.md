https://www.tensorflow.org/extend/tool_developers/


tf.GraphDef对应graph.proto定义

生成graph_def有两种方式，文本和二进制的


	from google.protobuf import text_format
	graph_def = graph_pb2.GraphDef()

	if FLAGS.input_binary:
	    graph_def.ParseFromString(f.read())
	else:
	    text_format.Merge(f.read(), graph_def)

从graph.proto可以看到最重要的属性就是graph_def.node包含了所有的计算节点
每个节点都是NodeDef，有对应的属性name, op, input, device, attr

freezing_graph.py把graph_def和checkpoint合并到一起，生成一个包含定义和权重的文件，去掉多余的信息，把variable变成const，一个文件直接给生产使用
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

a TensorProto object from a NodeDef representing a Const op by calling something like some_node_def.attr['value'].tensor
const类型的得NodeDef，通过some_node_def.attr['value'].tensor，可以获得对应的TensorProto对象
比如这个，global_step原来是个变量，在pb里是个const
不知道为什么，我从graph.pbtxt里读出来的，并不是fronzed的mobel里读出来，为什么variable也是const？

	In [38]: n
	Out[38]:
	name: "global_step/Initializer/zeros/shape_as_tensor"
	op: "Const"
	attr {
	  key: "_class"
	  value {
	    list {
	      s: "loc:@global_step"
	    }
	  }
	}
	attr {
	  key: "_output_shapes"
	  value {
	    list {
	      shape {
		dim {
		}
	      }
	    }
	  }
	}
	attr {
	  key: "dtype"
	  value {
	    type: DT_INT32
	  }
	}
	attr {
	  key: "value"
	  value {
	    tensor {
	      dtype: DT_INT32
	      tensor_shape {
		dim {
		}
	      }
	    }
	  }
	}


挨个看下运行时目录下所有的文件都是干什么的

- checkpoint
- tfevents
- graph.pbtxt
- ckpt
	data
	index
	meta
- pipeline


checkpoint                                                  model.ckpt-42097.index
events.out.tfevents.1523614409.huanghaodeMacBook-Pro.local  model.ckpt-42097.meta
graph.pbtxt                                                 model.ckpt-42197.data-00000-of-00001
model.ckpt-41909.data-00000-of-00001                        model.ckpt-42197.index
model.ckpt-41909.index                                      model.ckpt-42197.meta
model.ckpt-41909.meta                                       model.ckpt-42295.data-00000-of-00001
model.ckpt-42003.data-00000-of-00001                        model.ckpt-42295.index
model.ckpt-42003.index                                      model.ckpt-42295.meta
model.ckpt-42003.meta                                       pipeline.config
model.ckpt-42097.data-00000-of-00001


	In [45]: len(graph_def.node)
	Out[45]: 47898

	In [48]: opcnt = defaultdict(int)
	In [49]: for n in graph_def.node: opcnt[n.op] += 1

	In [52]: sorted(opcnt.items(), key=lambda i: i[1], reverse=1)[:10]
	Out[52]:
	[('Const', 15984),
	 ('Identity', 3668),
	 ('Sub', 2318),
	 ('Mul', 2177),
	 ('Fill', 1559),
	 ('Assign', 1374),
	 ('Reshape', 1347),
	 ('Switch', 1166),
	 ('Add', 1132),
	 ('IsVariableInitialized', 845)]

