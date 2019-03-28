
import tensorflow as tf

from models.base import BaseModel
from utils import yolo
import colorsys


GOLDEN_RATIO = 0.618033988749895

def generate_colors(n, max_value=255):
    colors = []
    h = 0.1
    s = 0.5
    v = 0.95
    for i in range(n):
        h = 1 / (h + GOLDEN_RATIO)
        colors.append([c*max_value for c in colorsys.hsv_to_rgb(h, s, v)])

    return colors

class YoloBaseModel(BaseModel):
	"""Yolo base model class."""

	_checkpoint_path = None
	_names_path = None
	_anchors = None
	labels = None

	def __init__(self, input_shape):
		self._meta_graph_location = self._checkpoint_path+'.meta'
		self._input_shape = input_shape

		self._score_threshold = 0.3
		self._iou_threshold = 0.4
		self._sess = None
		self._raw_inp = None
		self._raw_out = None
		self._eval_inp = None
		self._eval_ops = None

		self.colors = None

	def _evaluate(self, matrix):
		# TODO: We can merge normalization with other OPs, but we need to
		# redefine input tensor for this. Anyway this works faster then
		# normalizing input data with python or openCV or numpy.
		normalized = self._sess.run(self._raw_out,
									feed_dict={self._raw_inp: matrix})
		return self._sess.run(self._eval_ops,
							  feed_dict={self._eval_inp: normalized})

	def init(self):
		if bool(self.labels) == bool(self._names_path):
			raise AttributeError(
				'Model must define either "labels" or "names path" not both.')

		if self._names_path:
			with open(self._names_path) as f:
				self.labels = f.read().splitlines()

		if not self._anchors:
			raise AttributeError('Model must define "_anchors".')

		self._sess = tf.Session()
		self.colors = generate_colors(len(self.labels))

		saver = tf.train.import_meta_graph(
			self._meta_graph_location, clear_devices=True,
			import_scope='evaluation'
		)
		saver.restore(self._sess, self._checkpoint_path)

		eval_inp = self._sess.graph.get_tensor_by_name('evaluation/input:0')
		eval_out = self._sess.graph.get_tensor_by_name('evaluation/output:0')
		
		with tf.name_scope('normalization'):
			raw_inp = tf.placeholder(tf.float32, self._input_shape,
									 name='input')
			inp = tf.image.resize_images(raw_inp, eval_inp.get_shape()[1:3])
			inp = tf.expand_dims(inp, 0)
			raw_out = tf.divide(inp, 255., name='output')

		with tf.name_scope('postprocess'):
			outputs = yolo.head(eval_out, self._anchors, len(self.labels))
			self._eval_ops = yolo.evaluate(
				outputs, self._input_shape[0:2],
				score_threshold=self._score_threshold,
				iou_threshold=self._iou_threshold)

		self._raw_inp = raw_inp
		self._raw_out = raw_out
		self._eval_inp = eval_inp

		self._sess.run(tf.global_variables_initializer())

	def close(self):
		self._sess.close()

	def evaluate(self, matrix):
		objects = []
		for box, score, class_id in zip(*self._evaluate(matrix)):
			top, left, bottom, right = box
			objects.append({
				'box': {
					'top': top,
					'left': left,
					'bottom': bottom,
					'right': right
				},
				'score': score,
				'class': class_id,
				'class_name': self.labels[class_id],
				'color': self.colors[class_id]
			})
		return objects


class Yolo2Model(YoloBaseModel):

	_checkpoint_path = 'data/yolo2/yolo_model.ckpt'
	_names_path = 'data/yolo2/yolo2.names'
	_anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],
				[7.88282, 3.52778], [9.77052, 9.16828]]
