import os
import os.path
import cv2
import glob
import imutils
import numpy as np
from matplotlib import pyplot as plt




CAPTCHA_IMAGE_FOLDER = 'data'
OUTPUT_FOLDER = 'extracted'
# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}


for (i, captcha_image_file) in enumerate(captcha_image_files):
	print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

	# Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"), grab the base filename as the text
	filename = os.path.basename(captcha_image_file)
	captcha_correct_text = os.path.splitext(filename)[0]
	image = cv2.imread(captcha_image_file)
	kernel = np.ones((3, 3), np.uint8)
	height = image.shape[1]
	width = image.shape[0]
	img = cv2.medianBlur(image, 3)
	
	print(img.shape)
	img_dil = cv2.dilate(img, kernel, iterations=1)
	img_erode = cv2.erode(img_dil, kernel, iterations=1)
#     img = cv2.medianBlur(image,3)
#     img_dil = cv2.dilate(img,kernel,iterations = 1)
#     img_erode = cv2.erode(img_dil,kernel,iterations = 1)
	gray = cv2.cvtColor(img_erode, cv2.COLOR_BGR2GRAY)

	# Add some extra padding around the image
#     gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

	# threshold the image (convert it to pure black and white)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#     img_erode = cv2.erode(thresh,kernel,iterations = 1)
	# find the contours (continuous blobs of pixels) the image
	contours = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Hack for compatibility with different OpenCV versions
	contours = contours[1] if imutils.is_cv3() else contours[0]

	letter_image_regions = []

	# Now we can loop through each of the four contours and extract the letter inside of each one
	for contour in contours:
#         # Get the rectangle that contains the contour
		(x, y, w, h) = cv2.boundingRect(contour)

		# Compare the width and height of the contour to detect letters that are conjoined into one chunk
		if h*w > 300:
			letter_image_regions.append((x, y, w, h))

#     # If we found more or less than 4 letters in the captcha, skip the image 
	if len(letter_image_regions) != 5:
		print(captcha_image_file)
		continue

# Sort the detected letter images based on the x coordinate to make sure we are processing them from left-to-right so we match the right image with the right letter
	print(len(letter_image_regions))
	letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

#     # Save out each letter as a single image
	for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
#         # Grab the coordinates of the letter in the image
		x, y, w, h = letter_bounding_box

#         # Extract the letter from the original image with a 2-pixel margin around the edge
		letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

#         # Get the folder to save the image in
		save_path = os.path.join(OUTPUT_FOLDER, letter_text)

#         # if the output directory does not exist, create it
		if not os.path.exists(save_path):
			os.makedirs(save_path)

#         # write the letter image to a file
		count = counts.get(letter_text, 1)
		p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
		cv2.imwrite(p, letter_image)

#         # increment the count for the current key
		counts[letter_text] = count + 1
