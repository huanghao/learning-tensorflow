https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

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
  extract_sub_graph
  

output_graph
output_node_names

clear_devices
initializer_nodes
variable_names_whitelist
variable_names_blacklist

checkpoint_version: saver_pb2.SaverDef.V1 V2

