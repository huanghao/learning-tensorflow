https://www.tensorflow.org/programmers_guide/saved_model

tf.train.Saver 在graph里加上了一些save和restore op，对所有变量，或者指定的一些变量。
Saver提供了一些方法来执行这些op，指定checkpoint的位置

tf使用二进制的checkpoint文件保存变量名字和他们对应的tensor的值

tf的model文件也是代码，小心不安全的代码。使用Tensorflow Securely

model.ckpt只是一系列文件的前缀

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

    Model saved in path: /tmp/model.ckpt

    ➜  tt ll /tmp/model.ckpt.*
    -rw-r--r--  1 huanghao  wheel    32B Apr 17 14:59 /tmp/model.ckpt.data-00000-of-00001
    -rw-r--r--  1 huanghao  wheel   143B Apr 17 14:59 /tmp/model.ckpt.index
    -rw-r--r--  1 huanghao  wheel   4.5K Apr 17 14:59 /tmp/model.ckpt.meta

构造Saver的时候，指定需要保存的变量列表

- 可以创建任意多个saver对象，来分别保存和恢复模型的不同子集。 相同的变量可以被用在不同的saver，它的值会在restore()调用的时候变更
- 如果只有部分变量从restore来，剩下的部分就需要使用intializer来初始化
- 查看一个checkpoint中的变量，可以使用inspect_checkpoint库，特别是print_tensors_in_checkpoint_file函数
- 默认情况下，saver使用tf.Varaible.name，但你也可以手动指定名字

	$ python -m tensorflow.python.tools.inspect_checkpoint --file_name /tmp/model.ckpt
	v1 (DT_FLOAT) [3]
	v2 (DT_FLOAT) [5]

	$ python -m tensorflow.python.tools.inspect_checkpoint --file_name model.ckpt-42197 | wc -l
	     588

tf.saved_model.simple_save(
	session,
	export_dir,
	inputs={'x': x, 'y': y},
	outputs={'z': z})
这种最简单的方法，可以被`Tensorflow serving`和`Predict API`支持

使用tf.saved_model.builder.SavedModelBuilder来手动创建SavedModel。可以用来保存多个MetaGraphDef
`MetaGraph` 是一个dataflow graph数据流图，和关联的变量，资产？？和签名
`MetaGraphDef`是pb格式的MetaGraph
`signature`是图中的input和output集合

每个MetaGraphDef保存的时候，还可以添加一些用户指定的tags。一般用来指定用途，或者硬件特性

assets可以在第一个MetaGraphDef添加的时候提供

tensorflow.python.tools.saved_model_cli 可以用来show和run
run的时候，input可以是.npy, .npz 和pickle格式

目录结构
- asserts/
  包含外部文件，例如词典
- assets.extra/
- variables/
    从Saver拿到的数据
    variables.data-????-of-????
    variables.index
saved_model.pb | saved_model.pbtxt
  图定义

http://stackabuse.com/tensorflow-save-and-restore-models/
----

Saver主要关注变量的保存，ckpt这套

SavedModel是一种新的方式，加入几个新的概念，比如Signature，和 Assets
signature指的是图的输入和输出
assets可以包含外部文件，用在初始化的时候

