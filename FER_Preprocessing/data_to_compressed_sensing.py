import glob
from skimage import io
import matplotlib.pyplot as plt
import cv2

import numpy as np
import os
import argparse
from PIL import Image


def magnitude_spectrum(image):
    """Resize an image to the given size."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    (w, h) = fshift.shape
    half_w,half_h = int(w/2),int(h/2)
    n = 1
    fshift[half_w-n:half_w+n+1,half_h-n:half_h+n+1]=0
    hiu = np.fft.ifft2(fshift).real
    #magnitude_spectrum = 20*np.log10(np.abs(fshift))


    return hiu


def magnitude_spectrum2images(input_dir, output_dir):
    """Resize the images in 'input_dir' and save into 'output_dir'."""
    for idir in os.scandir(input_dir):
        if not idir.is_dir():
            continue
        if not os.path.exists(output_dir + '/' + idir.name):
            os.makedirs(output_dir + '/' + idir.name)
        images = os.listdir(idir.path)
        n_images = len(images)
        for iimage, image in enumerate(images):
            try:
                with open(os.path.join(idir.path, image), 'r+b') as f:
                    with Image.open(f).convert("L") as img:
                        img = magnitude_spectrum(img)
                        cv2.imwrite(os.path.join(output_dir + '/' + idir.name, image), img)
                        #img.save(os.path.join(output_dir + '/' + idir.name, image), img.format)
            except(IOError, SyntaxError) as e:
                pass
            if (iimage + 1) % 1000 == 0:
                print("[{}/{}] fourier transformed images and saved into '{}'."
                      .format(iimage + 1, n_images, output_dir + '/' + idir.name))


def main(args):
    input_dir = 'D:/data/FER/RAF/basic/Image/aligned'
    output_dir = 'D:/data/FER/RAF/basic/Image/aligned_liu'
    magnitude_spectrum2images(input_dir, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='/run/datasets/VQA/Images',
                        help='directory for input images (unresized images)')

    parser.add_argument('--output_dir', type=str, default='/run/datasets/VQA/Resized_Images',
                        help='directory for output images (resized images)')


    args = parser.parse_args()

    main(args)



# file = 'C:/Users/1315/Desktop/test/happy_man.jpg'
# img=cv2.imread(file)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# fig = plt.figure(figsize=(12,8))
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# phase = np.angle(fshift)
#
# plt.subplot(131),plt.imshow(img,cmap='gray')
# plt.title('Input Image'),plt.xticks([]),plt.yticks([])
# plt.subplot(132),plt.imshow(magnitude_spectrum,cmap='gray')
# plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
# plt.subplot(133),plt.imshow(phase,cmap='gray')
# plt.title('Phase Spectrum'),plt.xticks([]),plt.yticks([])
#
# plt.savefig('a',dpi=300,bbox_inches = 'tight')
#
# plt.show()
#
# fig = plt.figure(figsize=(8,8))
# (w,h) = fshift.shape
# half_w,half_h = int(w/2),int(h/2)
#
# n = 15
# fshift[half_w-n:half_w+n+1,half_h-n:half_h+n+1]=0
# plt.imshow((20*np.log10(0.1+fshift)).astype(int),cmap='gray')
# plt.axis('off')
# plt.savefig('highfrequencyfilter',dpi=300,bbox_inches='tight')
# plt.show()