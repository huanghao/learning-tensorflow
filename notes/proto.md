https://developers.google.com/protocol-buffers/docs/overview
----

对比xml，小，快速，简单。可以向前兼容老的格式。

uniquely numbered field
optional / required / repeated field

2和3不兼容，我现在安装的版本是3.5.1
libprotoc 3.5.1

https://developers.google.com/protocol-buffers/docs/pythontutorial
----

The .proto file starts with a package declaration, which helps to prevent naming conflicts between different projects. In Python, packages are normally determined by directory structure, so the package you define in your .proto file will have no effect on the generated code. However, you should still declare one to avoid name collisions in the Protocol Buffers name space as well as in non-Python languages.

proto中的包名用来避免名字冲突。在python中，包名直接通过文件夹来判断，所以这个定义对python生成代码来说没有作用。但是为了其他语言也能够正常使用还是要写上。

The " = 1", " = 2" markers on each element identify the unique "tag" that field uses in the binary encoding. Tag numbers 1-15 require one less byte to encode than higher numbers, so as an optimization you can decide to use those tags for the commonly used or repeated elements, leaving tags 16 and higher for less-commonly used optional elements. Each element in a repeated field requires re-encoding the tag number, so repeated fields are particularly good candidates for this optimization.

数字是每个元素在二进制表示中的唯一标签。1-15比更大的数字会更节省空间。repeat字段是更好的选择。

optional可以设置default值。否则系统会自动初始化zero value。跟python的零值有点像

repeated字段，也包括空

python生成的代码，用metaclass来生成对应的访问函数，而不是直接生成

AttributeError设置不存在的字段，TypeError字段类型错误

这个枚举生成的是Person.HOME，而不是Person.PhoneType.HOME
	message Person {
	  enum PhoneType {
	    MOBILE = 0;
	    HOME = 1;
	    WORK = 2;
	  }

通用方法：
- IsInitialized()
- __str__()
- CopyFrom()
- Clear()
- SerializeToString(): 生成的二进制
- ParseFromString(data)

Protocol Buffers and O-O Design: You should never add behaviour to the generated classes by inheriting from them. 
用组合，别用继承

为了保持新老兼容，几条原则：
1）不要修改已存在字段的tag number
2）不要增加或者删除required字段
3）可以删除optional和repeated字段
4）可以增加新的optional或者repeated字段，但是需要使用新的tag number（即使已经被删除的字段使用过也不行）
老代码会忽略新的字段。被删除的optional字段会获得默认值，被删除的repeated字段为空。
新代码可以透明的读老格式的消息。新增的optional字段也是默认值。如果是repeated字段，空的情况可能来自于老代码，也可能是新代码设置为空。

反射，通过Message interface来实现。

https://developers.google.com/protocol-buffers/docs/proto
----

scalar types:
	int32, int64, uint32, uint64, sint32, sint64
	fixed32, fixed64, sfixed32, sfixed64：4字节，8字节bytes
	bool
	double, float
	string
	bytes
composite types:
	enum

tag一旦开始使用，就不应该再被修改
1-15只需要一个字节
16-2047需要两个字节
记着要留一些空间给未来会出现的常用字段。但是只能是optional的

1-2^29 - 1
19000 - 19999保留

历史原因，老的repeated字段需要写[packed=true]，为了高效。新版不用。

reserved 可以用来保留一些tag和name，主要是给删除的字段用的，防止后来的人重新使用这些名字
名字可以被重用吗？

编码细节
https://developers.google.com/protocol-buffers/docs/encoding#structure

import的目录怎么算?
运行protoc的时候，--proto_path/-I指定了import的目录，如果没有的话，就是当前目录
所以一般运行protoc需要在特定的目录下，保持目录结构
也可以指定多个目录

import public，老的proto文件变位置了，放一个指针一样

updating a message type:
- 不要修改tag
- 新字段必须是optional或者repeated
- 非required字段可以被删除，只要它的tag不被使用。添加前缀`OBSOLETE_`，或者reserved，可能会更好一点
- 非required字段可以被转成extension，或者相反，只要tag不变
- int32, uint32, int64, uint64和bool互相兼容。意味着这几种类型可以互相修改，而保持向前向后兼容。如果超过了范围，相当于截断。
- sint32, sint64 相互兼容，而与其他整数类型不兼容
- string, bytes 相互兼容，只要bytes存储的是utf8
- embedded message和bytes 兼容，只要bytes存储的是这个message的编码
- fixed32, sfixed32, fixed64, sfixed64 兼容
- optional和repeated兼容
- 修改default值一般来说都是可以的。记住default值不会被放到message里传递。接收端收到消息，发现需要default的时候，去读default，而不会看发送端的代码里的default是什么。也就是说default只在接收端起作用。
- enum和int32, int64, uint32, uint64在二进制格式上兼容，但是客户端在解码消息的时候可能会区别对待。不存在的enum会被丢弃，会让`has_`返回false，而return返回第一个enum或者default。如果是repeated enum，不合法的值会被丢弃。从int升级为enum的时候，要注意超过范围的情况
- 当前的java和c++实现中，不认识的enum会被丢弃。导致一个data被同一个client编码再解码的时候仍然会被认识，这个奇怪的行为
- 把一个optional 改成一个新的 oneof 是安全的。把多个optional 合成一个新的 oneof 的时候可能是安全，但是你需要保证多个字段只有一个被设置。把任何字段移到一个已经存在的 oneof 字段都是不安全的。

extension类似于继承，在基类里留好tag的范围，在子类里定义需要的字段，使用这个范围的tag
extension字段可以是除了 oneof 和 map 之外的任何类型
两个不同的子类不能使用相同的tag

oneof 和 optional 有点类似，但是更省内存，所有的字段共享相同的内存，只有一个会出现。有点类似于c的union。设置一个字段，会自动清空其他字段的值

map 的 key 可以是string 或者 int，任何scalar（除了float和bytes）都可以作为key。
value 可以是任何类型，但不能是另一个map
map无序，但是在生成txt格式的proto文件的时候，key是排序显示的
解码时有相同的key，后一个取胜。如果从txt文件中读取，重复的key会报错

service，定义rpc接口，可以自动生成stub
使用gRPC

完整的option列表在google/protobuf/descriptor.proto
文件级别的option和message级别的option，还有field级别的

protoc生成代码的时候，还可以直接生成zip或者jar

https://developers.google.com/protocol-buffers/docs/proto3
----

Any

https://developers.google.com/protocol-buffers/docs/style
----

message, enum, service 名字使用驼峰
字段使用下划线
枚举值使用全大写下划线
rpc方法名使用驼峰

https://developers.google.com/protocol-buffers/docs/encoding
----


