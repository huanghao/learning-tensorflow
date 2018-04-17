类型定义只有25个proto，可以仔细看一下，基本就明白存储和交换的所有message类型了
	$ ls core/framework/*.proto | wc -l
	      25

graph.proto
  repeated NodeDef node
  --> node_def.proto
    string name, op, device, repeated string input
    map<string, AttrValue> attr
    --> attr_value.proto
      value 可以是多种类型，string/int/float/bool/type/shape/tensor/list(...)/func/placeholder
      list(...)又可以是多种类型, list(string/int/float/bool/type/shape/tensor/attr)
      --> tensor.proto
        dtype, shape
        bytes tensor_content 序列化之后的原始tensor内容
        repeated int/float/bytes/bool/DT_xx xx_val 类型相关的表示法，都是repeated，可就是可选，可能是把上面tensor_content转换一下表现形式
      --> tensor_shape.proto
        repeated dim { int size, string name }
      --> types.proto
        enum DataType { DT_* }
      --> resource_handle.proto：tf资源的句柄。句柄可以被序列化和反序列化，在一次run中使用，在不同的executions之间是不合法的
        string device, container, name, maybe_type_name; int hash_code
  VersionDef
  --> versions.proto：一段序列化数据的版本
    int producer, min_consumer; repeated int bad_consumers
    一个消费者在满足以下几个条件的时候，才能消费这段数据：
    1）producer >= min_producer
    2) consumer >= min_consumer
    3) consumer not in bad_consumers
  FunctionDefLibrary：暂时还只是实验字段
  --> function.proto：library是一组命名文件的集合
    repeated FunctionDef：
      OpDef signature 函数的名字，参数，返回值，和属性
      --> op_def.proto
        string name
        repeated ArgDef input_arg { string name, description, type_attr, number_attr, type_list_attr; bool is_ref; DataType type}
        repeated ArgDef output_arg
      map<string, AttrValue> attr
      repeated NodeDef
      map<string, string> ret
    repeated GradientDef
      string function_name 函数名
      string gradient_func 梯度函数名

tensor_slice.proto：用来表示一个tensor的切片
  repeated Extent{ int start, int length } 相当于每个维度里的一个范围
tensor_description.proto：用来描述一个tensor。和上面的tensor有啥区别？这里有个地址，是否就指向了上面的tensor。另外，地址应该只能在本进程中使用
  dtype, shape
  AllocationDescription
  --> allocation_description.proto
    int requested_bytes（请求的字节） allocated_bytes（分配的字节）
    string allocator_name（使用的allocator的名字）
    bool has_single_reference（这个tensor是否只有一个引用？？）
    uint64 ptr（分配的地址）

kernel_def.proto
  string op, device_type; repeated string host_memory_arg, label repeated AttrConstraint { string name, AttrValue allowed_values } remote_fused_graph_execute_info.proto GraphDef remote_graph repeated string graph_input_node_name, graph_output_node_name string executor_name summary.proto：一组用来显示的名字和值。在训练过程中周期产生
  oneof value {float/bytes/Image/HistogramProto/Audio/TensorProto}

variable.proto
  string variable_name, initial_value_name, initializer_name（初始化op的名字）, snapshot_name
  bool is_resource
  SaveSliceInfoDef

cost_graph.proto：存储所有的node对应的内存，和计算时间开销
step_stats.proto：存储了按step对应的内存，和计算开销
log_memory.proto：应该是用来记录内存分配历史的


device_attributes.proto：设备信息
  string name, device_type
  int memory_limit
  DeviceLocality {int bus_id, numa_node, incarnation; string physical_device_desc; LocalLinks { repeated InterconnectLink { device_id, type, strength }}}

api_def.proto
graph_transfer_info.proto
iterator.proto
reader_base.proto


DataType

DT_INVALID
DT_FLOAT/DOUBLE
DT_INT8/UINT8/QINT8/INT16/QINT16/UINT16/QUINT16/INT32/QINT32/INT64/UINT64
DT_STRING
DT_COMPLEX64/COMPLEX128
DT_HALF
DT_RESOURCE
DT_VARIANT

DT_xx_REF：指向前面的类型


meta_graph.proto: 包含图的元信息，GraphDef是图本身
  MetaInfoDef：元信息的元信息。。。
    meta_graph_version
    repeated string tags: train, serve, gpu, tpu, etc
    string tensorflow_version 写这个图的tf版本
  GraphDef
  SaverDef
  --> saver.proto
  map<string, CollectionDef>
    oneof NodeList/BytesList/Int64List/FloatList/AnyList
  TensorInfo
    oneof string name/CooSparse; dtype; TensorShapeProto
  map<string, SignatureDef>
    map<string, TensorInfo> inputs, outputs
    string method_name
  repeated AssetFileDef
    TensorInfo; string filename
