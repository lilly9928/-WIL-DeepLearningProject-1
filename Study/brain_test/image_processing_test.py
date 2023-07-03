import numpy as np
from PIL import Image
import pydicom
import matplotlib.pyplot as plt
import cv2

def fourier(img):
    height, width = img.size
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    row, col = int(height / 2), int(width / 2)
    LPF = np.zeros((height, width, 2), np.uint8)
    LPF[row - 50:row + 50, col - 50:col + 50] = 1
    LPF_shift = dft_shift * LPF
    LPF_ishift = np.fft.ifftshift(LPF_shift)
    LPF_img = cv2.idft(LPF_ishift)
    LPF_img = cv2.magnitude(LPF_img[:, :, 0], LPF_img[:, :, 1])
    out = 20*np.log(cv2.magnitude(LPF_img[:, :, 0], LPF_img[:, :, 1]))

    inverse_shift = np.fft.fftshift(dft_shift)
    inverse_dft = cv2.dft(inverse_shift, flags=cv2.DFT_INVERSE)
    out2 = cv2.magnitude(inverse_dft[:, :, 0], inverse_dft[:, :, 1])

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('original')
    plt.subplot(132)
    plt.imshow(out, cmap='gray')
    plt.title('dft')
    plt.subplot(133)
    plt.imshow(out2, cmap='gray')
    plt.title('inverse')

    plt.show()

def defourier(img):

    inverse_shift = np.fft.fftshift(img)
    img_back = cv2.idft(inverse_shift)
    # inverse_dft = cv2.dft(inverse_shift, flags=cv2.DFT_INVERSE)
    out2 = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('original')
    plt.subplot(133)
    plt.imshow(out2, cmap='gray')
    plt.title('inverse')

    plt.show()


# test_crop_src = 'D:/data/brain/preprocessed_image_data/preprocessed_data/fourier_cut_cropped_img/cropped_img/ANO_0001_032_(98,303,171,353)_SAH.jpg'
# img=Image.open(test_crop_src)
#
# defourier(img)

test_crop_src = 'D:/data/brain/preprocessed_image_data/preprocessed_data/cropped_img/ANO_0001_032_(98,303,171,353)_SAH.npy'
test_whole_src ='D:/data/brain/preprocessed_image_data/preprocessed_data/whole_img/ANO_0001_032.npy'
np_array_c = np.load(test_crop_src)
np_array_w = np.load(test_whole_src)
np_array_no=np_array_w
np_array_w=np_array_w/4095 *255
pil_image=Image.fromarray(np_array_c)
fourier(pil_image)

#pil_image.show()

import pydicom
import matplotlib.pyplot as plt

dcm = pydicom.read_file('D:/data/brain/raw_image_data/ANO_0001/G0301_20220621/0002_20220621_062450/IN_00032_0000355771.dcm')
array=dcm.pixel_array
plt.imshow(dcm.pixel_array ,cmap=plt.cm.bone)
plt.show()