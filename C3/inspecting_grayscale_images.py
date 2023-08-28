import os
import cv2
import matplotlib.pyplot as plt

curr_dir = os.path.realpath(os.path.dirname(__file__))

img = cv2.imread(os.path.join(curr_dir, "Hemanvi.jpeg"))
# Crop image
img = img[50:250, 40:240]
# Convert image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Show image
plt.imshow(img_gray, cmap="gray")
plt.show()

img_gray_small = cv2.resize(img_gray, (25, 25))
plt.imshow(img_gray_small, cmap="gray")
plt.show()
