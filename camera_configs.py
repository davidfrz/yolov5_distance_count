import cv2
import numpy as np

right_camera_matrix = np.array([[394.0102, -0.1445, 292.2634],
                                [0, 393.0441, 249.9139],
                                [0., 0, 1.0000]])
right_distortion = np.array([[-0.0051, 0.0329, 0.0030, -0.0049, 0]])

left_camera_matrix = np.array([[393.2362, -0.8477, 316.5539],
                               [0, 393.5102, 249.9189],
                               [0., 0, 1.0000]])
left_distortion = np.array([[-0.0164, 0.0569, 0.0043, -0.0034, 0]])

R = np.matrix([
    [1.0000, 0.0012, -0.0014],
    [-0.0012, 1.0000, -0.0051],
    [0.0014, 0.0051, 1.0000],
])

T = np.array([-114.3589, 0.0010, 2.9948])


size = (640, 480) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)