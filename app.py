import cv2
from main import *
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
	model_path = './yolo.h5'
	save_path = './yolo_model/yolo.ckpt'
	saver, graph, model_input, is_training, model_output = load_model(model_path)
	class_names = read_classes("./coco_classes.txt")
	anchors = read_anchors("./yolo_anchors.txt")
	with graph.as_default():
		model_output = postprocess_output_1(model_output, anchors, len(class_names))
		init = tf.global_variables_initializer()			
		with tf.Session(config= tf.ConfigProto(allow_soft_placement=True)) as sess:
			init.run()
			saver.restore(sess=sess, save_path=save_path)
			cap = cv2.VideoCapture(0)
			out = cv2.VideoWriter('output.mp4', -1, 10.0, (640,480))

			while(cap.isOpened()):
				ret, frame = cap.read()
				if ret==True:
					image, _, _, _ = predict(sess, frame, model_output, model_input, is_training, class_names)
					image = np.array(image)[...,[2,1,0]]
					out.write(image)
					cv2.imshow('Image',image)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
				else:
					break

			cap.release()
			out.release()
			cv2.destroyAllWindows()
