import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys
import matplotlib.pyplot as plt

def load_model(model_path = './yolo.h5'):
	tf.keras.models.load_model(compile=True, filepath=model_path, custom_objects=None)
	graph = tf.get_default_graph()
	saver = tf.train.Saver()
	model_input = graph.get_tensor_by_name('input_1:0')
	is_training = graph.get_tensor_by_name('batch_normalization_1/keras_learning_phase:0')
	model_output = graph.get_tensor_by_name('conv2d_23/BiasAdd:0')
	return saver, graph, model_input, is_training, model_output
	
def preprocess_image(image, model_size):
	image = np.array(image.resize(model_size), dtype=np.float32)
	norm_image = image/255.
	norm_image = np.expand_dims(norm_image,0)
	return norm_image

def postprocess_output_1(output, anchors, num_classes):
	num_anchors = len(anchors)
	anchors_tensor = tf.reshape(tf.Variable(anchors,dtype='float32'), [1, 1, 1, num_anchors, 2])
	conv_dims = tf.shape(output)[1:3]
	conv_height_index = tf.keras.backend.arange(0, stop=conv_dims[0])
	conv_width_index = tf.keras.backend.arange(0, stop=conv_dims[1])
	conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])
	conv_width_index = tf.tile(tf.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
	conv_width_index = tf.keras.backend.flatten(tf.transpose(conv_width_index))
	conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]))
	conv_index = tf.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
	conv_index = tf.cast(conv_index, output.dtype)

	output = tf.reshape(output, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
	conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), output.dtype)
	box_confidence = tf.sigmoid(output[..., 4:5])
	box_xy = tf.sigmoid(output[...,:2])
	box_wh = tf.exp(output[...,2:4])
	box_class_probs = tf.nn.softmax(output[...,5:])
	box_xy = (box_xy + conv_index) / conv_dims
	box_wh = box_wh * anchors_tensor / conv_dims
	
	return box_confidence, box_xy, box_wh, box_class_probs
	
def boxes_to_corners(box_xy, box_wh):
	mins = box_xy -(box_wh/2.)
	maxes = box_xy + (box_wh / 2.)
	
	return tf.keras.backend.concatenate([
        tf.convert_to_tensor(mins[..., 1:2]),   
        tf.convert_to_tensor(mins[..., 0:1]),
        tf.convert_to_tensor(maxes[..., 1:2]),
        tf.convert_to_tensor(maxes[..., 0:1])
    ])
    
def filter_boxes(box_confidence, boxes, box_class_probs,threshold = 0.6):
	box_scores = box_confidence * box_class_probs
	box_classes = tf.argmax(box_scores, axis = -1)
	box_class_scores = tf.reduce_max(box_scores,axis=-1)
	filtering_mask = box_class_scores >= threshold
	scores = tf.boolean_mask(mask=filtering_mask,tensor=box_class_scores)
	boxes = tf.boolean_mask(mask=filtering_mask,tensor=boxes)
	classes = tf.boolean_mask(mask=filtering_mask,tensor=box_classes)
	return scores, boxes, classes
	
def nms(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
	max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
	tf.variables_initializer([max_boxes_tensor]).run()
	nms_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores, iou_threshold=iou_threshold,
												max_output_size = max_boxes_tensor)
	scores = tf.gather(scores,nms_indices)
	boxes = tf.gather(boxes, nms_indices)
	classes = tf.gather(classes, nms_indices)
	return scores, boxes, classes
	
def scale_boxes(boxes, image_shape):
	height, width = image_shape[0], image_shape[1]
	image_dims = tf.stack([height, width, height, width])
	image_dims = tf.reshape(image_dims,[1,4])
	boxes = boxes * image_dims
	return boxes
	
def postprocess_output_2(model_output, image_shape = (720., 1280.), max_boxes = 10, score_threshold = 0.6,iou_threshold = 0.5):
	box_confidence,box_xy,box_wh, box_class_probs = model_output
	boxes = boxes_to_corners(box_xy, box_wh)
	
	scores, boxes, classes = filter_boxes(box_class_probs = box_class_probs, box_confidence = box_confidence,
										boxes = boxes,threshold = score_threshold)
										
	boxes = scale_boxes(boxes, image_shape)
	scores, boxes, classes = nms(boxes = boxes, classes = classes, iou_threshold = iou_threshold,max_boxes = max_boxes, scores=scores)
	return scores,boxes, classes
	
def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
    font = ImageFont.truetype("Verdana.ttf",14)
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.seed(49)
    np.random.shuffle(colors)
    np.random.seed(None)
    return colors

def predict(sess,image,model_output,model_input, model_learning_phase,class_names):
	image = image[...,[2,1,0]]
	image = Image.fromarray(image,"RGB")
	image_data = preprocess_image(image, model_size = (608,608))
	image_shape = tf.Variable(image.size[::-1], dtype=tf.float32)
	tf.variables_initializer([image_shape]).run()
	out_scores, out_boxes, out_classes = postprocess_output_2(sess.run(model_output, feed_dict={model_input:image_data, model_learning_phase:0}), image_shape=image_shape)
	out_scores, out_boxes, out_classes = sess.run([out_scores, out_boxes, out_classes])
	colors = generate_colors(class_names)
	draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
	return image, out_scores, out_boxes, out_classes
	
	
