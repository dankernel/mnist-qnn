
import sys
import numpy as np
import cv2
import qnn_utils

from enum import Enum
from termcolor import colored, cprint

np.set_printoptions(threshold=sys.maxsize)

NUMBER_LINE = '├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤'

# Option 
use_ReLU = True

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

def get_scale(l: list, num_bits: int):

    temp_max = np.max(l)
    temp_min = np.min(l)
    scale = (temp_max - temp_min) / (2 ** num_bits)

    return scale


def quantization(path: str, num_bits: int=8, use_zp: bool=False):
    """
    Quantization

    :param str path: ndarray file path
    :param int num_bits: bits
    :param bool use_zp: use zero point
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
    scale = (temp_max - temp_min) / (2 ** (num_bits - 1))

    # Print Debug
    if __debug__:
        print_debug('L1', temp_min, temp_max, scale)

    if use_zp is True:
        zero_point = 0 -(temp_min // scale)
        zero_point = zero_point.astype(np.int8)
    else:
        zero_point = 0

    """
    Encoding zero point
    example)
    [-3, -1,  5] -> +3(-min) -> [3, 2, 8]
    [ 2,  4,  5] -> -2(-min) -> [0, 2, 3]
    """
    quantized_tensor = (tensor // scale) + zero_point
    quantized_tensor = quantized_tensor.astype(np.int8)
    return tensor, quantized_tensor, scale, zero_point

def _matmul(a, b):

    if __debug__:
        a_shape = a.shape
        b_shape = b.shape
        if a.ndim <= 1:
            a_shape = (1, a.shape[0])
        if b.ndim <= 1:
            b_shape = (1, b.shape[0])
        c_shape = (a_shape[0], b_shape[1])
        print('A   shape :{} dtype: {}'.format(a_shape, a.dtype))
        print('B   shape :{} dtype: {}'.format(b_shape, b.dtype))
        print('         {0:5}               {1:5}               {2:5}'.format(a_shape[1], b_shape[1], c_shape[1]))
        print('      ┌──────┐         ┌─────────┐         ┌─────────┐')
        print('      │      │         │         │         │         │')
        print('{0:5} │      │ * {1:5} │         │ = {2:5} │         │'.format(a_shape[0], b_shape[0], c_shape[0]))
        print('      │      │         │         │         │         │')
        print('      └──────┘         └─────────┘         └─────────┘')
        print('')
        print('A :', a[:5], '...')
        print('B :', b[0][:5], '...')

    # matmul
    ret = np.matmul(a, b)

    if __debug__:
        print('C   shape :{} dtype: {}'.format(ret.shape, ret.dtype))
        print('min : {} max : {}'.format(min(ret), max(ret)))
        print('C :', ret[:5])

    # Get scale
    temp_scale = get_scale([min(ret), max(ret)], 8)
    print('temp_scale :', temp_scale, 1/temp_scale)

    print(colored('FC1 out', 'green', 'on_yellow'), colored(ret[:5], 'cyan'))
    ret = ret / temp_scale
    print(colored('* scale', 'green', 'on_yellow'), colored(ret[:5], 'cyan'))
    ret = ret.astype(int)
    print(colored('ast int', 'green', 'on_yellow'), colored(ret[:5], 'cyan'))

    return ret

def inference(path: str, inference_mode=None):

    use_zp = False
    inference_scale_resize = True

    inp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    inp = inp.reshape(784)
    inp = inp.astype(np.int32)
    print('inp shape :', inp.shape)

    # Load tensor
    fc1w, quantized_fc1w, fc1w_scale, fc1w_zp = quantization('mnist_dkdk_FP32_20170708_v1/FC1.npy', use_zp=use_zp)
    fc2w, quantized_fc2w, fc2w_scale, fc2w_zp = quantization('mnist_dkdk_FP32_20170708_v1/FC2.npy', use_zp=use_zp)
    fc3w, quantized_fc3w, fc3w_scale, fc3w_zp = quantization('mnist_dkdk_FP32_20170708_v1/FC3.npy', use_zp=use_zp)

    if __debug__:
        print('fc1w (FP32)   :', colored(fc1w[0][:5], 'red'))
        print('fc1w (INT8)   :', colored(quantized_fc1w[0][:5], 'red'))
        print('fc1w scale    :', colored(fc1w_scale, 'red'))
        print('fc2w scale    :', colored(fc2w_scale, 'red'))
        print('fc3w scale    :', colored(fc3w_scale, 'red'))
        print('fc1w zp       :', colored(fc1w_zp, 'red'))
        print('fc1w - zp     :', colored((quantized_fc1w[0][:5] - fc1w_zp), 'red'))
        print('fc1w - zp * s :', colored((quantized_fc1w[0][:5] - fc1w_zp) * fc1w_scale, 'red'))

    if inference_mode == Inference.INT8:

        qnn_utils.ndarray_to_bin(quantized_fc1w, './bin/FC1.bin')
        qnn_utils.ndarray_to_bin(quantized_fc2w, './bin/FC2.bin')
        qnn_utils.ndarray_to_bin(quantized_fc3w, './bin/FC3.bin')

        # zero point calibration (decoding)
        fc1w = quantized_fc1w
        fc2w = quantized_fc2w
        fc3w = quantized_fc3w

        if use_zp:
            fc1w -= fc1w_zp
            fc2w -= fc2w_zp
            fc3w -= fc3w_zp

    # FC1
    temp = _matmul(inp, fc1w)
    temp = np.maximum(0, temp)

    # FC2
    temp = _matmul(temp, fc2w)
    temp = np.maximum(0, temp)
    
    # FC3
    temp = _matmul(temp, fc3w)
    
    print(temp)

    from scipy.special import softmax
    temp = softmax(temp)
    print(temp)
    result = np.argmax(temp)
    print(result)

    return result

if __name__ == '__main__':


    test_type = Inference.INT8 # Inference.INT8 or Inference.FP32

    for i in range(1):
        ret = inference('test_image/{}.png'.format(i), inference_mode=test_type) 
        print(colored('{} {}'.format(ret, i), 'blue'))
        assert ret == i

    print(ret)

