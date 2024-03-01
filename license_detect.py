# First, make sure to install the necessary packages using pip:
# !pip install easyocr numpy matplotlib imutils

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import easyocr

# Read the image using OpenCV
image = cv2.imread('5.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply bilateral filtering to reduce noise while keeping edges sharp
filter = cv2.bilateralFilter(gray, 11, 17, 17)

# Detect edges in the image using Canny edge detection
edge = cv2.Canny(filter, 30, 200)

# Find contours in the edge image
ext_count = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(ext_count)

# Sort the contours based on area in descending order and select top 10 contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    # Approximate the contour to a polygon with less vertices
    apprx = cv2.approxPolyDP(contour, 10, True)
    if len(apprx) == 4:  # License plate usually has 4 vertices
        location = apprx
        break

# Create a mask to extract the license plate region
msk = np.zeros(gray.shape, np.uint8)
if location is not None:
    extracted_plate = cv2.drawContours(msk, [location], 0, 255, -1)
    extracted_plate = cv2.bitwise_and(image, image, mask=msk)

    # Get the coordinates of the license plate region
    (x, y) = np.where(msk == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))

    # Crop the license plate region from the grayscale image
    final_plate = gray[x1:x2+1, y1:y2+1]

    # Initialize the EasyOCR reader with Bengali and English language support
    read_char = easyocr.Reader(['bn', 'en'])

    # Use EasyOCR to detect text in the license plate region
    detected = read_char.readtext(final_plate)
    detected_texts = [region_info[1] for region_info in detected]
    print(detected_texts)
else:
    print("No license plate detected")
    exit()
