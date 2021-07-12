
import cv2
import numpy as np

inp = cv2.imread('test_image/0.png', cv2.IMREAD_GRAYSCALE)
inp = inp.reshape(-1)
np.save('./image.npy', inp)

