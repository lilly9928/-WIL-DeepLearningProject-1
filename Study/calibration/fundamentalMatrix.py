import cv2
import numpy
import matplotlib

orb = cv2.ORB()

frame1 = cv2.imread('C:/Users/1315/Desktop/clean/cal/2/img_1_00013.jpg')
frame2 = cv2.imread('C:/Users/1315/Desktop/clean/cal/2/img_2_00000.jpg')


detector = cv2.ORB_create()
kp1, des1 = detector.detectAndCompute(frame1, None)
kp2, des2 = detector.detectAndCompute(frame2, None)

des1, des2 = map(numpy.float32, (des1, des2))

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good, pts1, pts2 = [], [], []

# Ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        good.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = numpy.float32(pts1)
pts2 = numpy.float32(pts2)

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

print(F)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(numpy.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img1, _ = drawlines(frame1, frame2, lines1, pts1, pts2)

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, F)
lines2 = lines2.reshape(-1, 3)
img2, _ = drawlines(frame2, frame1, lines2, pts2, pts1)

matplotlib.pyplot.subplot(121)
matplotlib.pyplot.imshow(img1)
matplotlib.pyplot.subplot(122)
matplotlib.pyplot.imshow(img2)
matplotlib.show()