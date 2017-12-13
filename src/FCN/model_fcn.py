from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import math

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
# import BatchDatsetReader as dataset
import DatsetReader as dataset
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


class Model(object):
	def __init__(self):
		self.input = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input')
		self.annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='annotation')
		self.keep_probability = tf.placeholder(tf.float32, name='keep_probability')
		self.build_model()

		config = tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False
		)
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)

		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			print("Model restored...")

	def build_model(self):
		with slim.arg_scope(
				[slim.conv2d, slim.fully_connected],
				activation_fn=tf.nn.relu,
				weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
				weights_regularizer=slim.l2_regularizer(0.0005)
		):
			net = slim.repeat(self.input, 2, slim.conv2d, 64, [3, 3], scope='conv1')
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
		relu_dropout6 = tf.nn.dropout(relu6, keep_prob=self.keep_probability)

		W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
		b7 = utils.bias_variable([4096], name="b7")
		conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
		relu7 = tf.nn.relu(conv7, name="relu7")
		relu_dropout7 = tf.nn.dropout(relu7, keep_prob=self.keep_probability)

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

		shape = tf.shape(self.input)
		deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
		W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
		b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
		conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

		annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

		self.logits = conv_t3
		self.anno_pred = tf.expand_dims(annotation_pred, dim=3)

		self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=self.logits,
			labels=tf.squeeze(self.annotation, squeeze_dims=[3]),
			name="entropy"
		)))

		self.trainable_var = tf.trainable_variables()

		self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss, var_list=self.trainable_var)
		return

	def run_single_step(self, image, annotations):
		_, train_loss, prediction, logits, summary_str = self.sess.run(
			[self.train_op, self.loss, self.anno_pred, self.logits, self.summary_op],
			feed_dict={
				self.input: image,
				self.annotation: annotations,
				self.keep_probability: 0.85
			}
		)
		return train_loss, prediction, logits, summary_str

	def eval_single_step(self, image, annotations):
		eval_loss, eval_pred = self.sess.run(
			[self.loss, self.anno_pred],
			feed_dict={
				self.input: image,
				self.annotation: annotations,
				self.keep_probability: 1.0
			}
		)
		return eval_loss, eval_pred

	def _accuracy(self, annotations, prediction):
		annotations = np.squeeze(annotations, axis=3)
		prediction = np.squeeze(prediction, axis=3)
		sum_annotation = 0
		sum_pred_annotation = 0
		for index in range(FLAGS.batch_size):
			sum_annotation += np.sum(annotations[index])

			rows, cols = np.shape(prediction[index])
			for i in range(rows):
				for j in range(cols):
					if prediction[index][i, j] == 1 and annotations[index][i, j] == 1:
						sum_pred_annotation += 1

		acc = float(sum_pred_annotation) / sum_annotation

		return acc

	def train(self, train_dataset, eval_dataset):
		tf.summary.scalar('loss', self.loss)
		self.summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, self.sess.graph)
		for itr in range(MAX_ITERATION):
			# input images by batch
			image, annotations = train_dataset.next_batch(FLAGS.batch_size)
			train_loss, prediction, logits, summary_str = self.run_single_step(image, annotations)

			if itr % 10 == 0:
				# acc = self._accuracy(annotations, prediction)

				print("%s ---> Step: %d, Train_loss: %g" % (datetime.datetime.now(), itr, train_loss))
				with open(os.path.join(FLAGS.logs_dir, 'train_log.txt'), 'a') as f:
					f.write("%s ---> Step: %d, Train_loss: %g" % (datetime.datetime.now(), itr, train_loss) + '\n')
				summary_writer.add_summary(summary_str, itr)

			if itr % 100 == 0 and itr != 0:
				# load validation dataset
				eval_images, eval_annotations = eval_dataset.next_batch(FLAGS.batch_size)
				eval_loss, eval_pred = self.eval_single_step(eval_images, eval_annotations)

				# acc = self._accuracy(eval_annotations, eval_pred)
				with open(os.path.join(FLAGS.logs_dir, 'train_log.txt'), 'a') as f:
					f.write("%s --->  Step: %d, Validation_loss: %g" % (datetime.datetime.now(), itr, eval_loss) + '\n')
				print("%s --->  Step: %d, Validation_loss: %g" % (datetime.datetime.now(), itr, eval_loss))
				self.saver.save(self.sess, FLAGS.logs_dir + "model.ckpt", itr)

	def eval(self, train_dataset, eval_dataset):
		num_iter = int(math.ceil(float(FLAGS.data_number) / FLAGS.batch_size))

		for number in range(num_iter):
			if FLAGS.subset == "train":
				images_to_check, annotation_to_check = train_dataset.next_batch(FLAGS.batch_size)
			elif FLAGS.subset == "validation":
				images_to_check, annotation_to_check = eval_dataset.get_random_batch(FLAGS.batch_size)

			_, eval_pred = self.eval_single_step(images_to_check, annotation_to_check)

			# acc = self._accuracy(annotation_to_check, eval_pred)

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
						eval_pred[itr].astype(np.uint8) * 255.0,
						FLAGS.logs_dir,
						name="pred_" + str(number)
					)
					print("Saved image: %d" % number)


def main(_):
	train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
	print(len(train_records))
	print(len(valid_records))

	image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
	train_dataset_reader = dataset.BatchDatset(train_records, image_options)
	eval_dataset_reader = dataset.BatchDatset(valid_records, image_options)

	model = Model()
	if FLAGS.mode == 'train':
		model.train(train_dataset_reader, eval_dataset_reader)
	elif FLAGS.mode == 'validation':
		model.eval(train_dataset_reader, eval_dataset_reader)


if __name__ == '__main__':
	tf.app.run()
