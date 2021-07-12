
import sys
import numpy as np
import cv2
from enum import Enum

np.set_printoptions(threshold=sys.maxsize)

NUMBER_LINE = '├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤'

class Inference(Enum):
    FP32 = 1
    INT8 = 2

debug_option = {'print_all_layer': True}

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """

    return np.exp(x) / np.sum(np.exp(x), axis=0)

def print_debug(layer: str, min: float, max: float, scale: float):
    """
    Print debug 

    :param str layer: layer name
    :param float min:
    :param float max:
    :param float scale:
    :return: None
    """

    print('|-----------|---------|---------|---------|---------|---------|---------|')
    print('|   Layer   |   min   |   max   |  scale  | 1/scale | min * S | max * S |')
    print('|-----------|---------|---------|---------|---------|---------|---------|')
    print('| {0:9} | {1:7.2} | {2:7.2} | {3:7.2} | {4:7.4} | {5:7.4} | {6:7.4} |'.format(layer, min, max, scale, 1 / scale, min / scale, max / scale))
    print('|-----------|---------|---------|---------|---------|---------|---------|')

    print('')
    print('┏ {0: <30}{2: >30} ┒'.format(min, 0, max))
    print(NUMBER_LINE)
    print('')

    quantized_min = round(min / scale)
    quantized_max = round(max / scale)
    print('┏ {0: <30}{1: >30} ┒'.format(quantized_min, quantized_max))
    print(NUMBER_LINE)
    print('')

def quantization(path: str, num_bits: int=8):
    """
    Quantization

    :param str path: ndarray file path
    :param int num_bits: bits
    :returns: 
        - tensor - 
        - quantized_tensor - 
        - scale -
        - zero_point -
    """

    # FC1 Weight
    tensor = np.load(path)
    print(' [fc1w] Shape : {} / dtype : {}'.format(tensor.shape, tensor.dtype))

    # Max, Min, Scale
    temp_max = np.max(tensor)
    temp_min = np.min(tensor)
    scale = (temp_max - temp_min) / (2 ** (num_bits))

    # Print Debug
    if __debug__:
        print_debug('L1', temp_min, temp_max, scale)

    zero_point = 0 -(temp_min // scale)

    """
    Encoding zero point
    example)
    [-3, -1,  5] -> +3(-min) -> [3, 2, 8]
    [ 2,  4,  5] -> -2(-min) -> [0, 2, 3]
    """
    quantized_tensor = (tensor // scale) + zero_point
    quantized_tensor = quantized_tensor.astype(np.uint8)
    return tensor, quantized_tensor, scale, zero_point

def ndarray_to_bin(ndarray, out_path: str):
    """
    ndarray to bin file
    (4byte) dim
    (4byte) shape x dim

    :param ndarray: target numpy ndarrat
    :param str out_path: output path
    :return: None
    """

    with open(out_path, 'wb') as file:
        dim = len(ndarray.shape)
        file.write(dim.to_bytes(4, byteorder='little', signed=True))
        for s in range(dim):
            size = ndarray.shape[s]
            file.write(size.to_bytes(4, byteorder='little', signed=True))
        file.write(ndarray.tobytes())
 
def inference(path: str, inference_mode=None):

    inp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    inp = inp.reshape(-1)

    # Load tensor
    fc1w, quantized_fc1w, fc1w_scale, fc1w_zp = quantization('mnist_dkdk_FP32_20170708_v1/FC1.npy')
    fc2w, quantized_fc2w, fc2w_scale, fc2w_zp = quantization('mnist_dkdk_FP32_20170708_v1/FC2.npy')
    fc3w, quantized_fc3w, fc3w_scale, fc3w_zp = quantization('mnist_dkdk_FP32_20170708_v1/FC3.npy')

    if inference_mode == Inference.INT8:

        ndarray_to_bin(quantized_fc1w, './FC1.bin')
        ndarray_to_bin(quantized_fc2w, './FC2.bin')
        ndarray_to_bin(quantized_fc3w, './FC3.bin')
        
        # zero point calibration
        fc1w = quantized_fc1w - fc1w_zp
        fc2w = quantized_fc2w - fc2w_zp
        fc3w = quantized_fc3w - fc3w_zp

    temp = np.matmul(inp, fc1w)
    # temp = temp * fc1w_scale
    temp = np.maximum(0, temp)

    temp = np.matmul(temp, fc2w)
    # temp = temp * fc2w_scale
    temp = np.maximum(0, temp)
    
    temp = np.matmul(temp, fc3w)
    # temp = temp * fc3w_scale
    temp = temp * fc1w_scale * fc2w_scale * fc3w_scale

    print(temp)

    from scipy.special import softmax
    temp = softmax(temp)
    print(temp)
    result = np.argmax(temp)
    print(result)

    return result


if __name__ == '__main__':

    ret = inference('test_image/0.png', inference_mode=Inference.INT8)
    print(ret)

