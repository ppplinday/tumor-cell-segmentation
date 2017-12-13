from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import math

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
import tensorflow.contrib.slim as slim

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 32, "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_string("checkpoint_dir", "checkpoint_dir/", "model to restore and save")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', False, "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_integer('data_number', 30000, "number of data to evaluate")
tf.flags.DEFINE_string('subset', "validation", "validation of train, subset to evaluate")


MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 128


def fcn(inputs, keep_prob):
	with slim.arg_scope(
			[slim.conv2d, slim.fully_connected],
			activation_fn=tf.nn.relu,
			weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
			weights_regularizer=slim.l2_regularizer(0.0005)
	):
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		pool1 = slim.max_pool2d(net, [2, 2], scope='pool1')
		net = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		pool2 = slim.max_pool2d(net, [2, 2], scope='pool2')
		net = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		pool3 = slim.max_pool2d(net, [2, 2], scope='pool3')
		net = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		pool4 = slim.max_pool2d(net, [2, 2], scope='pool4')
		net = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		pool5 = slim.max_pool2d(net, [2, 2], scope='pool5')

	W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
	b6 = utils.bias_variable([4096], name="b6")
	conv6 = utils.conv2d_basic(pool5, W6, b6)
	relu6 = tf.nn.relu(conv6, name="relu6")
	relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

	W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
	b7 = utils.bias_variable([4096], name="b7")
	conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
	relu7 = tf.nn.relu(conv7, name="relu7")
	relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

	W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
	b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
	conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

	# now to upscale to actual image size
	deconv_shape1 = pool4.get_shape()
	W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
	b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
	conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(pool4))
	fuse_1 = tf.add(conv_t1, pool4, name="fuse_1")

	deconv_shape2 = pool3.get_shape()
	W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
	b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
	conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(pool3))
	fuse_2 = tf.add(conv_t2, pool3, name="fuse_2")

	shape = tf.shape(inputs)
	deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
	W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
	b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
	conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

	annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

	return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss, var_list):
	optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
	grads = optimizer.minimize(loss, var_list=var_list)
	return grads


def main(argv=None):
	keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
	image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
	annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

	pred_annotation, logits = fcn(image, keep_probability)
	# tf.summary.image("input_image", image, max_outputs=2)
	# tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
	# tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

	loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=logits,
		labels=tf.squeeze(annotation, squeeze_dims=[3]),
		name="entropy"
	)))
	tf.summary.scalar("entropy", loss)

	trainable_var = tf.trainable_variables()
	if FLAGS.debug:
		for var in trainable_var:
			utils.add_to_regularization_and_summary(var)
	train_op = train(loss, trainable_var)

	print("Setting up summary op...")
	summary_op = tf.summary.merge_all()

	print("Setting up image reader...")
	train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
	print(len(train_records))
	print(len(valid_records))

	print("Setting up dataset reader")
	image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
	if FLAGS.mode == 'train':
		train_dataset_reader = dataset.BatchDatset(train_records, image_options)
	validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

	config = tf.ConfigProto(allow_soft_placement=True,
	                        log_device_placement=False)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	print("Setting up Saver...")
	saver = tf.train.Saver()
	summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

	sess.run(tf.global_variables_initializer())
	ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("Model restored...")

	if FLAGS.mode == "train":
		for itr in xrange(MAX_ITERATION):
			train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
			feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

			sess.run(train_op, feed_dict=feed_dict)

			if itr % 10 == 0:
				train_loss, summary_str, train_pred, logits_eval = sess.run(
					[loss, summary_op, pred_annotation, logits], feed_dict=feed_dict)

				train_annotations = np.squeeze(train_annotations, axis=3)
				train_pred = np.squeeze(train_pred, axis=3)
				sum_annotation = 0
				sum_pred_annotation = 0
				for index in range(FLAGS.batch_size):
					sum_annotation += np.sum(train_annotations[index])

					rows, cols = np.shape(train_pred[index])
					for i in range(rows):
						for j in range(cols):
							if train_pred[index][i, j] == 1 and train_annotations[index][i, j] == 1:
								sum_pred_annotation += 1

				acc = float(sum_pred_annotation) / sum_annotation

				print("Step: %d, Train_loss: %g, ACC: %f" % (itr, train_loss, acc))
				with open(os.path.join(FLAGS.logs_dir, 'train_log.txt'), 'a') as f:
					f.write("Step: %d, Train_loss: %g, ACC: %f" % (itr, train_loss, acc) + '\n')
				summary_writer.add_summary(summary_str, itr)

			if itr % 500 == 0:
				valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
				feed_dict = {image: valid_images, annotation: valid_annotations, keep_probability: 1.0}
				valid_loss, valid_pred = sess.run(
					[loss, pred_annotation],
					feed_dict=feed_dict
				)
				valid_annotations = np.squeeze(valid_annotations, axis=3)
				valid_pred = np.squeeze(valid_pred, axis=3)

				sum_pred = 0
				sum_annotation = 0
				sum_pred_annotation = 0

				for index in range(FLAGS.batch_size):
					sum_pred += np.sum(valid_pred[index])
					sum_annotation += np.sum(valid_annotations[index])

					rows, cols = np.shape(valid_pred[index])
					for i in range(rows):
						for j in range(cols):
							if valid_pred[index][i, j] == 1 and valid_annotations[index][i, j] == 1:
								sum_pred_annotation += 1
				acc = float(sum_pred_annotation) / sum_annotation
				with open(os.path.join(FLAGS.logs_dir, 'train_log.txt'), 'a') as f:
					f.write(
						"%s ---> Validation_loss: %g     acc: %f" % (datetime.datetime.now(), valid_loss, acc) + '\n')
				print("%s ---> Validation_loss: %g     acc: %f" % (datetime.datetime.now(), valid_loss, acc))
				saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

	elif FLAGS.mode == "test":
		if FLAGS.subset == "train":
			train_dataset_reader = dataset.BatchDatset(train_records, image_options)

		num_iter = int(math.ceil(float(FLAGS.data_number) / FLAGS.batch_size))

		total_sum_annotation = 0
		total_sum_pred_annotation = 0

		for number in range(num_iter):

			sum_annotation = 0
			sum_pred_annotation = 0

			if FLAGS.subset == "validation":
				images_to_check, annotation_to_check = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
			elif FLAGS.subset == "train":
				images_to_check, annotation_to_check = train_dataset_reader.next_batch(FLAGS.batch_size)
			feed_dict = {image: images_to_check, annotation: annotation_to_check, keep_probability: 1.0}
			pred = sess.run(
				pred_annotation,
				feed_dict=feed_dict
			)
			annotation_to_check = np.squeeze(annotation_to_check, axis=3)
			pred = np.squeeze(pred, axis=3)

			for index in range(FLAGS.batch_size):
				sum_annotation += np.sum(pred[index])
				total_sum_annotation += np.sum(pred[index])
				rows, cols = np.shape(pred[index])
				for i in range(rows):
					for j in range(cols):
						if annotation_to_check[index][i, j] == 1 and pred[index][i, j] == 1:
							sum_pred_annotation += 1
							total_sum_pred_annotation += 1
			acc = float(sum_pred_annotation) / sum_annotation

			print("step:   " + str(number) + "               accuracy:    " + str(acc))

			# choose how many picture to show
			if number >= 0:
				for itr in range(FLAGS.batch_size):
					utils.save_image(
						images_to_check[itr].astype(np.uint8),
						FLAGS.logs_dir,
						name="inp_" + str(number)
					)
					utils.save_image(
						annotation_to_check[itr].astype(np.uint8) * 255.0,
						FLAGS.logs_dir,
						name="gt_" + str(number)
					)
					utils.save_image(
						pred[itr].astype(np.uint8) * 255.0,
						FLAGS.logs_dir,
						name="pred_" + str(number)
					)
					print("Saved image: %d" % number)

		total_acc = float(total_sum_pred_annotation) / total_sum_annotation

		print("total_acc:         " + str(total_acc))
		with open(os.path.join(FLAGS.logs_dir, 'eval_log.txt'), 'a') as f:
			f.write("number_data:       " + str(FLAGS.data_number) + '\n')
			f.write("test on:           " + str(FLAGS.subset) + '\n')
			f.write("total_acc:         " + str(total_acc) + '\n')


if __name__ == "__main__":
	tf.app.run()
