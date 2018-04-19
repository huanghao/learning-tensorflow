https://github.com/tensorflow/models/blob/master/research/object_detection/train.py
----

给定了pipeline config，用这个函数读
config_util.get_configs_from_pipeline_file(pipeline_config_path)

或者从三个文件读config
config_util.get_configs_from_multiple_files(
	model_config_path,
	train_config_path,
	train_input_config_path)

config的参考: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md

config里有几个部分：model，train_config, train_input_reader, eval_config, eval_input_reader，后两个跟eval相关的没有用？

- model_builder.build
- preprocessor_builder.build, data_argmentation_options
- model_deploy.DeploymentConfig
- create_input_queue
- _create_losses, regularization_losses
- model_deploy.create_clones, optimize_clones
- optimizer_builder.build, tf.train.SyncReplicasOptimizer
- trainig_optimizer.apply_gradients
- fine_tune_checkpoint
- slim.learning.train

objection_detection.train 准备了各种参数，最后调用了slim.learning.train
