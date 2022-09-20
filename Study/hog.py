#
# from skimage.io import imread
# from skimage.feature import hog
# import matplotlib.pyplot as plt
# import cv2
#
# img = cv2.imread('C:/Users/1315/Desktop/data/ck_val/ck_val0.jpg')
#
# #hog extraction
# _, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualize=True, channel_axis=True)
#
# #show image
# plt.imshow(img)
# plt.show()

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from skimage.io import imread
import cv2

src = cv2.imread('C:/Users/1315/Desktop/data/ck_train/ck_train745.jpg')
image= cv2.resize(src,(244,244))

fd, hog_image = hog(image, orientations=24, pixels_per_cell=(16,16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()