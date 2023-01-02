import cv2
import numpy as np
from scipy import ndimage

roberts_cross_v = np.array([[1, 0],
                            [0, -1]])

roberts_cross_h = np.array([[0, 1],
                            [-1, 0]])

# img = cv2.imread("result1.jpg", 0).astype('float64')
# img /= 255.0
# vertical = ndimage.convolve(img, roberts_cross_v)
# horizontal = ndimage.convolve(img, roberts_cross_h)
#
# edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
# edged_img *= 255
#cv2.imwrite("output1.jpg", edged_img)
src = cv2.imread("result2.jpg", cv2.IMREAD_GRAYSCALE)
# dx = cv2.Sobel(src, -1, 1, 0) # delta 값을 지정해주지 않으면 미분이 - 부분은 0
# dx = cv2.convertScaleAbs(dx)
# dy = cv2.Sobel(src, -1, 0, 1)
# dy = cv2.convertScaleAbs(dy)
# img_sobel = cv2.addWeighted(dx,1,dy,1,0)
# cv2.imwrite("output4.jpg", img_sobel)

# kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
# kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
# img_prewittx = cv2.filter2D(src, -1, kernelx)
# img_prewitty = cv2.filter2D(src, -1, kernely)

img_canny = cv2.Canny(src,100,200)

cv2.imwrite("output7.jpg",img_canny)