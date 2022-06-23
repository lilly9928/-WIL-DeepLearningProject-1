import numpy as np
import cv2

imageA = cv2.imread('C:/Users/1315/Desktop/cal/img_2_00002.jpg')  # 오른쪽 사진
imageB = cv2.imread('C:/Users/1315/Desktop/cal/img_1_00021.jpg')  # 왼쪽 사진

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kpA, desA = sift.detectAndCompute(grayA, None)
kpB, desB = sift.detectAndCompute(grayB, None)

bf = cv2.BFMatcher()
matches = bf.match(desA, desB)

sorted_matches = sorted(matches, key=lambda x: x.distance)
res = cv2.drawMatches(imageA, kpA, imageB, kpB, sorted_matches[:30], None, flags=2)

src = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape((-1, 1, 2))
dst = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape((-1, 1, 2))
H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

print('H',H)

before = []
for x in range(imageA.shape[1]):
    for y in range(imageA.shape[0]):
        point = [x, y, 1]
        before.append(point)
before = np.array(before).transpose()

after = np.matmul(H, before)
after = after / after[2, :]
after = after[:2, :]
after = np.round(after, 0).astype(np.int)

height, width, _ = imageA.shape
result = np.zeros((height, width * 2, 3), dtype=np.uint8)
for pt1, pt2 in zip(before[:2, :].transpose(), after.transpose()):
    if pt2[1] >= height:
        continue

    if np.sum(pt2 < 0) >= 1:
        continue

    result[pt2[1], pt2[0]] = imageA[pt1[1], pt1[0]]
result[0: height, 0: width] = imageB

cv2.imshow('result', result)

cv2.waitKey()
cv2.destroyAllWindows()