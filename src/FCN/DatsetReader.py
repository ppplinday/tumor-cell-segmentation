"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc


class BatchDatset:
	files = []
	images = []
	annotations = []
	image_options = {}
	batch_offset = 0
	epochs_completed = 0

	def __init__(self, records_list, image_options={}):
		"""
		Intialize a generic file reader with batching for list of files
		:param records_list: list of file records to read -
		sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
		:param image_options: A dictionary of options for modifying the output image
		Available options:
		resize = True/ False
		resize_size = #size of output image - does bilinear resize
		color=True/False
		"""
		print("Initializing Batch Dataset Reader...")
		print(image_options)
		self.files = np.asarray(records_list)
		self.image_options = image_options

	def _read_images(self, start, end):
		self.__channels = True
		self.images = np.array([self._transform(filename['image']) for filename in self.files[start:end]])
		self.__channels = False
		self.annotations = np.array(
			[np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files[start:end]])
		self.labels = np.array([filename['label'] for filename in self.files[start:end]])

		# print (self.images.shape)
		# print (self.annotations.shape)
		# print (self.labels)

	def _transform(self, filename):
		image = misc.imread(filename)
		if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
			image = np.array([image for i in range(3)])

		if self.image_options.get("resize", False) and self.image_options["resize"]:
			resize_size = int(self.image_options["resize_size"])
			resize_image = misc.imresize(image,
										 [resize_size, resize_size], interp='nearest')
		else:
			resize_image = image

		return np.array(resize_image)

	#def get_records(self):
	#	return self.images, self.annotations

	def reset_batch_offset(self, offset=0):
		self.batch_offset = offset

	def next_batch(self, batch_size):
		start = self.batch_offset
		self.batch_offset += batch_size
		if self.batch_offset > self.files.shape[0]:
			# Finished epoch
			self.epochs_completed += 1
			print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
			# Shuffle the data
			np.random.shuffle(self.files)
			# Start next epoch
			start = 0
			self.batch_offset = batch_size

		end = self.batch_offset
		self._read_images(start, end)
		return self.images, self.annotations

	def next_batch_label(self, batch_size):
		start = self.batch_offset
		self.batch_offset += batch_size
		if self.batch_offset > self.files.shape[0]:
			# Finished epoch
			self.epochs_completed += 1
			print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
			# Shuffle the data
			np.random.shuffle(self.files)
			# Start next epoch
			start = 0
			self.batch_offset = batch_size

		end = self.batch_offset
		self._read_images(start, end)
		return self.images, self.labels

	def _read_random_images(self, cnds):
		self.__channels = True
		self.images = np.array([self._transform(filename['image']) for filename in self.files[cnds]])
		self.__channels = False
		self.annotations = np.array(
			[np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files[cnds]])
		self.labels = np.array([filename['label'] for filename in self.files[cnds]])

		print (self.images.shape)
		print (self.annotations.shape)
		print (self.labels)
		
	def get_random_batch(self, batch_size):
		indexes = np.random.randint(0, self.files.shape[0], size=[batch_size]).tolist()
		self._read_random_images(indexes)
		return self.images, self.annotations
