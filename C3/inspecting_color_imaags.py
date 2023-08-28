import os
import cv2
import matplotlib.pyplot as plt

curr_dir = os.path.realpath(os.path.dirname(__file__))

img = cv2.imread(os.path.join(curr_dir, "Hemanvi.jpeg"))
img = img[50:250, 40:240, :]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
print(img.shape)

crop = img[-3:, -3:]
print(crop)
plt.imshow(crop)
plt.show()
