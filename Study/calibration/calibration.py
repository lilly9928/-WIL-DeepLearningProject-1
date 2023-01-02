
import cv2
import numpy as np
import math


img1 = cv2.imread('C:/Users/1315/Desktop/clean/cal/2/img_1_00013.jpg')
img2 = cv2.imread('C:/Users/1315/Desktop/clean/cal/2/img_2_00000.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

size = img1.shape
distortion_coeffs = np.zeros((4,1))
focal_length = size[1]
center = (size[1]/2, size[0]/2)
matrix_camera = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

figure_points_3D = np.array([
                            (10.0, 10.0*math.sqrt(2), 0.0),
                            (0.0, 0.0, 0.0),
                            (5.0, 30.0, 0.0),
                            (0.0, 30.0*math.sqrt(2), 0.0)
                        ] , dtype = "float32")

# ORB, BF-Hamming 로 knnMatch
detector = cv2.ORB_create()
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc1, desc2)

matches = sorted(matches, key=lambda x:x.distance)
good_matches = matches[:4]


print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))
print('# of matches:', len(matches))
print('# of good_matches:', len(good_matches))


# 근매칭점으로 원 변환 및 영역 표시
#src = image 1 , dst= image2
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2).astype(np.float32)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2).astype(np.float32)

src_pts_2d = np.array([ kp1[m.queryIdx].pt for m in good_matches ]).astype('float32')
dst_pts_2d = np.array([kp2[m.trainIdx].pt for m in good_matches ]).astype('float32')

print(src_pts_2d)
print(dst_pts_2d)

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(figure_points_3D, src_pts, gray1.shape[::-1],None,None)

#print('ret,mtx,dist,rvecs,tvecs',ret, mtx, dist, rvecs, tvecs)

success, s_vector_rotation, s_vector_translation = cv2.solvePnP(figure_points_3D,src_pts_2d, matrix_camera, distortion_coeffs, flags=0)
print('src_sucess',success)
if success:
    print('src_rotation',s_vector_rotation)
    print('src_translation', s_vector_translation)
    print('src_postion',s_vector_rotation*s_vector_translation)

d_success, d_vector_rotation, d_vector_translation = cv2.solvePnP(figure_points_3D,dst_pts_2d, matrix_camera, distortion_coeffs, flags=0)
print('dst_sucess',d_success)
if success:
    print('dst_rotation',d_vector_rotation)
    print('dst_translation', d_vector_translation)
    print('dst_postion', d_vector_rotation * d_vector_translation)


# RANSAC으로 변환 행렬 근사 계산
mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
h2, _ = cv2.findHomography(figure_points_3D, src_pts, cv2.RANSAC, 5.0)

#M = cv2.getPerspectiveTransform(src_pts,dst_pts)
#wrap_img = cv2.warpPerspective(img2,M,(img1.shape[1], img2.shape[0]))
im_dst = cv2.warpPerspective(img1, mtrx,(img1.shape[1], img2.shape[0]))
#im_dstt = cv2.warpPerspective(img2, h,(img1.shape[1], img2.shape[0]))
#wrap_img[0 : img1.shape[0], 0 : img2.shape[1]] = img2

h,w = img1.shape[:2]
pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
dst = cv2.perspectiveTransform(pts,mtrx)
if np.shape(mtrx) == ():
    print("No transformation possible")

# ## derive rotation angle from homography
theta = - math.atan2(mtrx[1, 0], mtrx[0, 0]) * 180 / math.pi



u, _, vh = np.linalg.svd(mtrx[0:2, 0:2])
R = u @ vh
angle = math.atan2(R[1,0], R[0,0])

print('rotation angle',theta)

print('rotation angle',angle)
# for m in matches:
#     print(kp1[m.queryIdx].pt,kp2[m.trainIdx].pt)

print('H1:',mtrx)
print('H2:',h)
print('H3:',h2)


# 정상치 매칭만 그리기
matchesMask = mask.ravel().tolist()

res2 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                    matchesMask = matchesMask,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


# 모든 매칭점과 정상치 비율 ---⑧
# accuracy=float(mask.sum()) / mask.size
# print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))

# 결과 출력
#cv2.imshow('Matching-All', res1)
cv2.imshow('Matching-Inlier ', res2)
cv2.imshow('dd ', im_dst)


cv2.waitKey()
cv2.destroyAllWindows()