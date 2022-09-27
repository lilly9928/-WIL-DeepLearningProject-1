import glob
from skimage import io
import matplotlib.pyplot as plt
import cv2

import numpy as np
import os
import argparse
from PIL import Image

file = 'C:/Users/1315/Desktop/data/ck_train/ck_train0.jpg'
img=cv2.imread(file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig = plt.figure(figsize=(12,8))
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
phase = np.angle(fshift)

plt.subplot(131),plt.imshow(img,cmap='gray')
plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum,cmap='gray')
plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(phase,cmap='gray')
plt.title('Phase Spectrum'),plt.xticks([]),plt.yticks([])

plt.savefig('a',dpi=300,bbox_inches = 'tight')

plt.show()

fig = plt.figure(figsize=(8,8))
(w,h) = fshift.shape
half_w,half_h = int(w/2),int(h/2)

n = 1
fshift[half_w-n:half_w+n+1,half_h-n:half_h+n+1]=0
plt.imshow((20*np.log10(0.1+fshift)).astype(int),cmap='gray')
plt.axis('off')
plt.savefig('highfrequencyfilter',dpi=300,bbox_inches='tight')
plt.show()

hiu = np.fft.ifft2(fshift).real
plt.figure(figsize=(10,10))
plt.subplot(111),plt.imshow(hiu,cmap='gray')
plt.title('Block'),plt.xticks([]),plt.yticks([])
plt.axis('off')
plt.savefig('hiu',dpi=300,bbox_inches='tight')
plt.show()

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
phase = np.angle(fshift)
fig = plt.figure(figsize=(8,8))
(w,h)=fshift.shape

n = 1
fshift[0:n,:] = 0
fshift[:,0:n]=0
fshift[w-n:w,:]=0
fshift[:,h-n:h]=0

plt.imshow((20*np.log10(0.1+fshift)).astype(int),cmap='gray')
plt.axis('off')
plt.savefig('lowfreqencyfiter',dpi=300,bbox_inches='tight')
plt.show()

liu = np.fft.ifft2(fshift).real
plt.figure(figsize=(10,10))
plt.subplot(111),plt.imshow(liu,cmap='gray')
plt.title('Blocked high frequency image'),plt.xticks([]),plt.yticks([])
plt.axis('off')
plt.savefig('liu',dpi=300,bbox_inches='tight')
plt.show()