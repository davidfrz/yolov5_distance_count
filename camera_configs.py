import cv2
import numpy as np

right_camera_matrix = np.array([[ 685.678224658699   ,-0.192637539609069,336.017417159595],
                               [  0 ,688.193435150549,210.716467543927],
                               [0., 0,1.0000]])
right_distortion = np.array([[-0.0435964411227556 ,0.0450419747474258 , 0.000198763680155245,0.00186508136997442,-0.794591605647541]])



left_camera_matrix = np.array([[ 693.750991126704   ,-0.0936854812002115  ,340.123299797090],
                                [   0,  696.629299099001 , 203.645820912582],
                                [0., 0, 1.0000]])
left_distortion = np.array([[-0.033255569027615 ,-0.146315104223902,-5.612051521043959e-04 ,6.775172462296378e-04,  -0.224045428301982]])

R = np.matrix([
    [  0.999971141990729 ,0.000140032394440041 ,0.00759576044163199],
    [ -0.000152508763838663  ,0.999998640304896 ,0.00164198947511729],
    [ -0.00759552018199586 ,-0.00164310051060516 ,0.999969803691030],
])

T = np.array([-16.339672559387495,0.342232831032356,-1.081531326484493])

size = (640, 480) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)