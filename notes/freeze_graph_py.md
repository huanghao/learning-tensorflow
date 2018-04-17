https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
----

从不同的参数里读出graph的定义，
然后restore（也可能根据参数来决定），
调用convert_variables_to_constants，
最后把output_graph_def写到文件里

输入主要是两个部分，graph和weights
graph可能从几个地方进来，input_saved_model_dir, input_meta_graph, input_graph
weights也可能从几个地方，input_saved_model_dir, input_checkpoint
如果是checkpoint的话，还有可能会附带一个input_saver
input_meta_graph里也会包含一个SaverDef

convert_variables_to_constants
  extract_sub_graph：指定了output node，找到那些能够到达终点的子图
    _bfs_for_reachable_nodes

  过一遍graph的所有node，如果node的op是 Variable, VariableV2, VarHandleOp
    检查黑白名单
    VarHandleOp，变量名就是 name + /Read/ReadVariableOp:0
  得到这些变量的值，run一下
  
  生成一个新的graphdef，循环之前的node，对应那些变量node，构造对应的const node
    对于ReadVariableOp类型的node，会直接用一个Identity node来代替
  copy 老graph的library
  

output_graph
output_node_names

clear_devices
initializer_nodes
variable_names_whitelist
variable_names_blacklist

checkpoint_version: saver_pb2.SaverDef.V1 V2

node的名字到底怎么组成的？
为什么变量经常要加上":0"这种形式？


https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py
----

参数：
- input_type: 输入node的类型，可以是`image_tensor`（默认），`encoded_image_string_tensor`，或者 `tf_example`
- input_shape: 对于image_tensor，指定shape，默认为[None, None, None, 3]
- * pipeline_config_path: TrainEvalPipelineConfig的文件地址
- * trained_checkpoint_prefix: ckpt的文件名前缀
- * output_directory
- config_override

pipeline_pb2.TrainEvalPipelineConfig

跑不通，出错，先放一下
