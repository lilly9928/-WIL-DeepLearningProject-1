
# import os
# import zipfile
# import numpy as np
# import torch
# import nibabel as nib
# from scipy import ndimage
# import torchvision
# def read_nifti_file(filepath):
#     """Read and load volume"""
#     # Read file
#     scan = nib.load(filepath)
#     # Get raw data
#     scan = scan.get_fdata()
#     return scan
#
#
# def normalize(volume):
#     """Normalize the volume"""
#     min = -1000
#     max = 400
#     volume[volume < min] = min
#     volume[volume > max] = max
#     volume = (volume - min) / (max - min)
#     volume = volume.astype("float32")
#     return volume
#
#
# def resize_volume(img):
#     """Resize across z-axis"""
#     # Set the desired depth
#     desired_depth = 64
#     desired_width = 128
#     desired_height = 128
#     # Get current depth
#     current_depth = img.shape[-1]
#     current_width = img.shape[0]
#     current_height = img.shape[1]
#     # Compute depth factor
#     depth = current_depth / desired_depth
#     width = current_width / desired_width
#     height = current_height / desired_height
#     depth_factor = 1 / depth
#     width_factor = 1 / width
#     height_factor = 1 / height
#     # Rotate
#     img = ndimage.rotate(img, 90, reshape=False)
#     # Resize across z-axis
#     img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
#     return img
#
#
# def process_scan(path):
#     """Read and resize volume"""
#     # Read scan
#     volume = read_nifti_file(path)
#     # Normalize
#     volume = normalize(volume)
#     # Resize width, height and depth
#     volume = resize_volume(volume)
#     return volume
#
#
# root = 'C:/Users/User/Desktop/'
# # Folder "CT-0" consist of CT scans having normal lung tissue,
# # no CT-signs of viral pneumonia.
# normal_scan_paths = [
#     os.path.join(root, "MosMedData/CT-0", x)
#     for x in os.listdir("C:/Users/User/Desktop/MosMedData/CT-0")
# ]
# # Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# # involvement of lung parenchyma.
# abnormal_scan_paths = [
#     os.path.join(root, "MosMedData/CT-23", x)
#     for x in os.listdir("C:/Users/User/Desktop/MosMedData/CT-23")
# ]
#
# print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
# print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))
#
#
# # Read and process the scans.
# # Each scan is resized across height, width, and depth and rescaled.
# abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
# normal_scans = np.array([process_scan(path) for path in normal_scan_paths])
#
# # For the CT scans having presence of viral pneumonia
# # assign 1, for the normal ones assign 0.
# abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
# normal_labels = np.array([0 for _ in range(len(normal_scans))])
#
# # Split data in the ratio 70-30 for training and validation.
# x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
# y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
# x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
# y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
# print(
#     "Number of samples in train and validation are %d and %d."
#     % (x_train.shape[0], x_val.shape[0])
# )
#
# import random
#
# from scipy import ndimage
#
#
# def rotate(volume):
#     """Rotate the volume by a few degrees"""
#
#     def scipy_rotate(volume):
#         # define some rotation angles
#         angles = [-20, -10, -5, 5, 10, 20]
#         # pick angles at random
#         angle = random.choice(angles)
#         # rotate volume
#         volume = ndimage.rotate(volume, angle, reshape=False)
#         volume[volume < 0] = 0
#         volume[volume > 1] = 1
#         return volume
#
#     #augmented_volume = tf.numpy_function(scipy_rotate, [volume], torch.float32)
#     augmented_volume = torchvision.transforms.functional.rotate([volume],scipy_rotate,torch.float32)
#     return augmented_volume
#
#
# def train_preprocessing(volume, label):
#     """Process training data by rotating and adding a channel."""
#     # Rotate volume
#     volume = rotate(volume)
#     volume =torch.unsqueeze(volume, axis=3)
#     return volume, label
#
#
# def validation_preprocessing(volume, label):
#     """Process validation data by only adding a channel."""
#     volume = torch.unsqueeze(volume, axis=3)
#     return volume, label
#
# # Define data loaders.
# train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
#
# batch_size = 2
# # Augment the on the fly during training.
# train_dataset = (
#     train_loader.shuffle(len(x_train))
#     .map(train_preprocessing)
#     .batch(batch_size)
#     .prefetch(2)
# )
# # Only rescale.
# validation_dataset = (
#     validation_loader.shuffle(len(x_val))
#     .map(validation_preprocessing)
#     .batch(batch_size)
#     .prefetch(2)
# )
#

import pydicom
import matplotlib.pyplot as plt
import numpy as np

img_array = np.load('D:/data/brain/preprocessed_data_v2/whole_img/ANO_0001_032.npy')

from PIL import Image

im = Image.fromarray(img_array)

print(img_array.shape)

dcm = pydicom.dcmread('D:/data/brain/raw_image_data/ANO_0001/G0301_20220621/0002_20220621_062450/IN_00001_0000355741.dcm')
print(dcm.size)
plt.imshow(dcm.pixel_array ,cmap=plt.cm.bone)
plt.show()