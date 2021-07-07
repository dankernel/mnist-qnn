
import sys
import numpy as np
import cv2
from scipy.special import softmax
np.set_printoptions(threshold=sys.maxsize)

def main(inp):
    """
    :param inp: 
    """

    inp = inp.reshape(-1)
    imp = 1 * (inp + 128)
    print('==inp==')
    print(inp.shape)
    print(inp.dtype)

    # FC1 Weight
    fc1w = np.load('bin/FC1.npy')
    fc1w = fc1w * 0.009713355451822281
    print('==fc1w==')
    print(fc1w.shape)
    print(fc1w.dtype)
    temp = np.matmul(fc1w, inp)
    print('==temp==')
    print(temp.shape)
    print(temp.dtype)

    # FC1 ReLU
    temp = np.maximum(0, temp)

    # FC2 Weight
    fc2w = np.load('bin/FC2.npy')
    fc2w = fc2w * 0.0044453018344938755
    print('==fc2w==')
    print(fc2w.shape)
    print(fc2w.dtype)
    temp = np.matmul(fc2w, temp)
    print('==temp==')
    print(temp.shape)
    print(temp.dtype)

    # FC2 ReLU
    temp = np.maximum(0, temp)

    # FC3 Weight
    fc3w = np.load('bin/FC3.npy')
    fc3w = fc3w * 0.004337742924690247
    print('==fc3w==')
    print(fc3w.shape)
    print(fc3w.dtype)
    temp = np.matmul(fc3w, temp)
    print('==temp==')
    print(temp.shape)
    print(temp.dtype)

    # FC3 Output
    print(temp)
    print(temp.dtype)

    # Softmax
    temp = softmax(temp)
    print(temp)

    # Output
    print(np.argmax(temp))

inp = cv2.imread('test_image/4.png', cv2.IMREAD_GRAYSCALE)
main(inp)


