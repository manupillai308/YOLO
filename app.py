import cv2
import tensorflow as tf

from models.yolo import Yolo2Model


def evaluate():
	win_name = 'Detector'
	cv2.namedWindow(win_name)
	cam = cv2.VideoCapture('./input.avi')

	source_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
	source_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)

	model = Yolo2Model(input_shape=(source_h, source_w, 3))
	model.init()
	try:
		while True:
			ret, frame = cam.read()

			predictions = model.evaluate(frame)

			for o in predictions:
				x1 = o['box']['left']
				x2 = o['box']['right']

				y1 = o['box']['top']
				y2 = o['box']['bottom']

				color = o['color']
				class_name = o['class_name']
				# Draw box
				cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

				# Draw label
				(test_width, text_height), baseline = cv2.getTextSize(
					class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
				cv2.rectangle(frame, (x1, y1),
							  (x1+test_width, y1-text_height-baseline),
							  color, thickness=cv2.FILLED)
				cv2.putText(frame, class_name, (x1, y1-baseline),
							cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)



			cv2.imshow(win_name, frame)

			key = cv2.waitKey(1) & 0xFF

			# Exit
			if key == ord('q'):
				break


	finally:
		cv2.destroyAllWindows()
		cam.release()
		model.close()


if __name__ == '__main__':
	evaluate()
