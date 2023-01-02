import cv2
import numpy as np
img = cv2.imread("C:/Users/1315/Desktop/image.jpg")
size = img.shape
image_points_2D = np.array([
                        (196, 141),  # Nose tip
                        (190, 202),  # Chin
                        (196, 124),  # Left eye corner
                        (236, 128),  # Right eye corner
                        (186, 175),  # Left mouth
                        (214, 177)   # Right mouth
                      ], dtype="double")

figure_points_3D = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

distortion_coeffs = np.zeros((4,1))
focal_length = size[1]
center = (size[1]/2, size[0]/2)
matrix_camera = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
success, vector_rotation, vector_translation = cv2.solvePnP(figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=0)
nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), vector_rotation, vector_translation, matrix_camera, distortion_coeffs)
for p in image_points_2D:
    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
point1 = ( int(image_points_2D[0][0]), int(image_points_2D[0][1]))

point2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

cv2.line(img, point1, point2, (255,255,255), 2)

cv2.imshow("Final",img)
cv2.waitKey(0)
cv2.destroyAllWindows()