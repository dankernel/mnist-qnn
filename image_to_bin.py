
import cv2
import numpy as np
import qnn_utils

inp = cv2.imread('test_image/0.png', cv2.IMREAD_GRAYSCALE)
print('shape :', inp.shape)
inp = inp.reshape(-1)
print('shape :', inp.shape)
print('inp[:200] :', inp[:200])

np.save('./image.npy', inp)
qnn_utils.ndarray_to_bin(inp, './image.bin')
print('[OK] saved!')
