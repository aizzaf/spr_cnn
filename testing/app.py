import time
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd


TF_LITE_MODEL_FILE_NAME = "model1.tflite"
interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

df = pd.read_csv('izzaspr.csv')
RA = df['RA (deg)'].tolist()[:38]
DE = df['DE (deg)'].tolist()[:38]
ID = df['Star ID'].tolist()[:38]
rotations = [0,120,144,168,192,216,24,240,264,288,312,336,48,72,96]

start = time.time()

image = cv2.imread('crop_translate/0_float32.jpg')
height, width = image.shape[:2]
image_input = cv2.resize(image, (256,256))
image_input = np.array(image_input, dtype='float32')

interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image_input,0))
interpreter.invoke()
label = interpreter.get_tensor(output_details[2]['index']).argmax()
rotation = interpreter.get_tensor(output_details[0]['index']).argmax()
regression = interpreter.get_tensor(output_details[1]['index'])
x = regression[0][0]
y = regression[0][1]

if interpreter.get_tensor(output_details[2]['index']).max() < 0.5:
	cv2.putText(image, 'Bintang belum dipelajari',(100,100), cv2.FONT_HERSHEY_TRIPLEX, 2, (255,255,255), thickness=2)
else:
	cv2.circle(image, (int(np.around(x*width+width/2)),int(np.around(y*height+height/2))), 98, (255,255,255), thickness=2)
	cv2.putText(image, 'ID=' + str(ID[label]) + ' RA=' + str(RA[label]) + '˚ DE=' + str(DE[label]) + '˚ ROLL=' + str(rotations[rotation]) + '˚', (100,100), cv2.FONT_HERSHEY_TRIPLEX, 2, (255,255,255), thickness=2)

end = time.time()

print(str(end-start)+' seconds')

image = cv2.resize(image, (800,600))
cv2.imshow('ta',image)

cv2.waitKey(0)

cv2.destroyAllWindows()
