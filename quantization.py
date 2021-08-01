
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

def print_debug_hex(a, b, c):


    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        print('Only 2 dim')
        return

    print('a', a.shape, a.dtype)
    print('b', b.shape, b.dtype)
    print('c', c.shape, c.dtype)

    a_uint8 = a.view(dtype=np.uint8)
    b_uint8 = b.view(dtype=np.uint8)
    c_uint8 = c.view(dtype=np.uint8)

    print('a', a_uint8.shape, a.dtype)
    print('b', b_uint8.shape, b.dtype)
    print('c', c_uint8.shape, c.dtype)

    a = a_uint8
    b = b_uint8
    c = c_uint8

    msg = []
    msg.append('                         xxx                            xxx                             xxx ')
    msg.append('     ┌──────────────────────┐       ┌──────────────────────┐        ┌──────────────────────┐')
    msg.append('     │ xx xx xx xx xx xx .. │       │ xx xx xx xx xx xx .. │        │ xx xx xx xx xx xx .. │')
    msg.append('     │ xx xx xx xx xx xx .. │ *     │ xx xx xx xx xx xx .. │ =      │ xx xx xx xx xx xx .. │')
    msg.append('     │ xx xx xx xx xx xx .. │       │ xx xx xx xx xx xx .. │        │ xx xx xx xx xx xx .. │')
    msg.append('     │ xx xx xx xx xx xx .. │       │ xx xx xx xx xx xx .. │        │ xx xx xx xx xx xx .. │')
    msg.append('     │ .. .. .. .. .. .. .. │       │ .. .. .. .. .. .. .. │        │ .. .. .. .. .. .. .. │')
    msg.append(' xxx └──────────────────────┘   xxx └──────────────────────┘    xxx └──────────────────────┘')

    val = [a, b, c]

    for row in range(len(msg)):
        msg[row] = msg[row].replace('xxx', '{:03}')

        for v in range(3):
            temp = val[v]
            for k in range(6):
                if 1 < row  and row - 2 < temp.shape[0] and k < temp.shape[1]:
                    msg[row] = msg[row].replace('xx', '{:02x}'.format(temp[row-2][k]), 1)
                else:
                    msg[row] = msg[row].replace('xx', '--', 1)

    print(msg[0].format(a.shape[1], b.shape[1], c.shape[1]))
    print(msg[1])
    print(msg[2])
    print(msg[3])
    print(msg[4])
    print(msg[5])
    print(msg[6])
    print(msg[7].format(a.shape[0], b.shape[0], c.shape[0]))
    """
    print(msg[0].format(a.shape[1], b.shape[1], c.shape[1]))
    print(msg[1])
    print(msg[2].format(a[0][0], a[0][1], a[0][2], b[0][0], b[0][1], b[0][2], c[0][0], c[0][1], c[0][2]))
    print(msg[3].format(a[1][0], a[1][1], a[1][2], b[1][0], b[1][1], b[1][2], c[1][0], c[1][1], c[1][2]))
    print(msg[4].format(a[2][0], a[2][1], a[2][2], b[2][0], b[2][1], b[2][2], c[2][0], c[2][1], c[2][2]))
    print(msg[5])
    print(msg[6].format(ashape[0], bshape[0], cshape[0]))
    """


def _matmul(a, b):

    print(a)

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
        print('                  {0:5}                      {1:5}               {2:5}'.format(a_shape[1], b_shape[1], c_shape[1]))
        print('      ┌───────────────┐         ┌────────────────┐         ┌─── ─── ─ ────┐')
        print('      │ {:02x} {:02x} {:02x}  ... │         │ {:02x} {:02x} {:02x}   ... │         │ {:02x} {:02x} {:02x} ... │'.format(10, 20, 30, 40, 50, 60, 100, 101, 102))
        print('      │ {:02x} {:02x} {:02x}  ... │ *       │ {:02x} {:02x} {:02x}   ... │ =       │ {:02x} {:02x} {:02x} ... │'.format(10, 20, 30, 10, 20, 30, 110, 111, 112))
        print('      │ {:02x} {:02x} {:02x}  ... │         │ {:02x} {:02x} {:02x}   ... │         │ {:02x} {:02x} {:02x} ... │'.format(11, 12, 13, 14, 15, 16, 120, 121, 122))
        print('      │ .. .. ..  ... │         │ .. .. ..   ... │         │ .. .. .. ... │'.format(11, 12, 13, 14, 15, 16))
        print(' {:4} └───────────────┘    {:4} └────────────────┘    {:4} └───────── ─ ──┘'.format(a_shape[0],  b_shape[0], c_shape[0]))
        print('')

        """
        for i in range(145, 155):
            print(i, a[i])
        """

    # matmul
    ret = np.matmul(a, b)
    print_debug_hex(a, b, ret)

    if __debug__:
        print('C   shape :{} dtype: {}'.format(ret.shape, ret.dtype))
        print('min : {} max : {}'.format(min(ret), max(ret)))
        print('C :', ret[:5])

    # Get scale
    temp_scale = get_scale([min(ret), max(ret)], 8)
    print('temp_scale :', temp_scale, 1/temp_scale)

    print(colored('FC1 out', 'green', 'on_yellow'), colored(ret[0][:5], 'cyan'))
    ret = ret / temp_scale
    print(colored('* scale', 'green', 'on_yellow'), colored(ret[0][:5], 'cyan'))
    ret = ret.astype(int)
    print(colored('ast int', 'green', 'on_yellow'), colored(ret[0][:5], 'cyan'))

    return ret

def inference(path: str, inference_mode=None):

    use_zp = False
    inference_scale_resize = True

    inp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    inp = inp.reshape(1, 784)
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

