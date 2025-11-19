import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.03, aruco_dict)
img = board.generateImage((600, 800))

cv2.imshow("ChArUco test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
