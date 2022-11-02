from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
import pylab
import cv2

pylab.rcParams['figure.figsize']=(8.0,10.0)

dataDir='D:/data/vqa/coco/simple_vqa'
dataType='val2014'
annFile='{}/Annotations/annotations/instances_{}.json'.format(dataDir,dataType)

coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
#
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['dog']);
imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#
# # load and display image
I = cv2.imread('D:/data/vqa/coco/simple_vqa/Images/%s/%s'%(dataType,img['file_name']))
# # use url to load image
#I = io.imread(img['coco_url'])
#
plt.imshow(I,cmap='gray')
plt.savefig("mygraph.png")
# #
# # load and display instance annotations
# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
# plt.savefig("mygraph1.png")
#
# # initialize COCO api for person keypoints annotations
# annFile = '{}/Annotations/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
# coco_kps=COCO(annFile)
#
# # load and display keypoints annotations
# plt.imshow(I); plt.axis('off')
# ax = plt.gca()
# annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco_kps.loadAnns(annIds)
# coco_kps.showAnns(anns)

# initialize COCO api for caption annotations
annFile = '{}/Annotations/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)

# load and display caption annotations
annIds = coco_caps.getAnnIds(imgIds=img['id']);
a=coco_caps.loadAnns(annIds)
anns = {coco_caps.loadAnns(annIds)[b]['caption']for b in range(len(a))}

print(anns)
#coco_caps.showAnns(anns)


