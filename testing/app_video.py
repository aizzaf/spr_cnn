import time
import numpy as np
import cv2
import tensorflow as tf
import imageio
import pandas as pd

images = []

df = pd.read_csv('izzaspr.csv')
RA = df['RA (deg)'].tolist()[:38]
DE = df['DE (deg)'].tolist()[:38]
ID = df['Star ID'].tolist()[:38]
rotations = [0,120,144,168,192,216,24,240,264,288,312,336,48,72,96]

TF_LITE_MODEL_FILE_NAME = "model1.tflite"
interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cam = cv2.VideoCapture(0)

if not cam.isOpened():
        sys.exit()

while (cam.isOpened()):

	start = time.time()

	check, frame = cam.read()
	height, width = frame.shape[:2]
	frame_input = cv2.resize(frame,(256,256))
	frame_input = np.array(frame_input,dtype='float32')

	interpreter.set_tensor(input_details[0]['index'], np.expand_dims(frame_input,0))
	interpreter.invoke()
	label = interpreter.get_tensor(output_details[2]['index']).argmax()
	rotation = interpreter.get_tensor(output_details[0]['index']).argmax()
	regression = interpreter.get_tensor(output_details[1]['index'])
	x = regression[0][0]
	y = regression[0][1]

	if interpreter.get_tensor(output_details[2]['index']).max() < 0.8:
		cv2.putText(frame, 'Bintang belum dipelajari',(5,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), thickness=1)
		print('bintang belum dipelajari')
	else:
		cv2.circle(frame, (int(np.around(x*width+width/2)),int(np.around(y*height+height/2))), 20, (255,255,255), thickness=1)
		cv2.putText(frame, 'ID=' + str(ID[label]) + ' RA=' + str(RA[label]) + 'deg DE=' + str(DE[label]) + 'deg ROLL=' + str(rotations[rotation]) + 'deg', (5,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), thickness=1)
		print('ID=' + str(ID[label]) + ' RA=' + str(RA[label]) + 'deg DE=' + str(DE[label]) + 'deg ROLL=' + str(rotations[rotation]) + 'deg')

	end = time.time()
	images.append(cv2.resize(frame,(820,616)))
	cv2.imshow('video', cv2.resize(frame,(820,616)))

	print(str(end-start)+' s')

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
imageio.mimsave('prediction_video.gif', images)
