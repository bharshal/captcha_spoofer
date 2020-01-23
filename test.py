from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import numpy as np
import pytesseract
import json
from keras.models import model_from_json

MODEL_FILENAME = "model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "data"
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(50,), replace=False)


def get_name(text):
	text_split = text.split('/')[1]
	get_name = text_split.split('.')[0]
	return get_name

with open(MODEL_LABELS_FILENAME, "rb") as f:
	lb = pickle.load(f)


json_file = open('model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(MODEL_FILENAME)
print("Loaded model from disk")



counter = 0

for image_file in captcha_image_files:
	kernel = np.ones((3, 3), np.uint8)
	image = cv2.imread(image_file)
	height = image.shape[1]
	width = image.shape[0]
	img = cv2.medianBlur(image, 3)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
	contours = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[1] if imutils.is_cv3() else contours[0]
	letter_image_regions = []
	for contour in contours:
		# Get the rectangle that contains the contour
			(x, y, w, h) = cv2.boundingRect(contour)
			if h*w > 400:
				letter_image_regions.append((x,y,w,h))
	if len(letter_image_regions) != 5:
		print(image_file)
		continue
	letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
	output = cv2.merge([gray] * 3)
	predictions = []
	for letter_bounding_box in letter_image_regions:
	# Grab the coordinates of the letter in the image
		x, y, w, h = letter_bounding_box

	# Extract the letter from the original image with a 2-pixel margin around the edge
		letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
#     print(letter_image.shape)

	# Re-size the letter image to 20x20 pixels to match training data
		letter_image = resize_to_fit(letter_image, 20, 20)

	# Turn the single image into a 4d list of images to make Keras happy
		letter_image = np.expand_dims(letter_image, axis=2)
		letter_image = np.expand_dims(letter_image, axis=0)

	# Ask the neural network to make a prediction
		prediction = model.predict(letter_image)

	# Convert the one-hot-encoded prediction back to a normal letter
		letter = lb.inverse_transform(prediction)[0]
		predictions.append(letter)
	actual_name = get_name(image_file)

	predicted_name = ''.join(predictions)

	if actual_name==str(predicted_name):
		counter = counter+1
	else:
		print("actual : " + actual_name)
		print("pred : " + predicted_name)

print(counter, "captchas predicted correctly out of 50.")




