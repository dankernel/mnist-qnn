
import sys
import numpy as np
import cv2
np.set_printoptions(threshold=sys.maxsize)

NUMBER_LINE = '├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤'

def print_debug(layer, min, max, scale):

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


def quantization(path, num_bits=8):

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

    # quantized_tensor = (tensor / scale).astype(np.int8)
    quantized_tensor = tensor // scale
    return tensor, quantized_tensor, scale

 
def main(path):

    inp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    inp = inp.reshape(-1)

    # Load tensor
    run_mode = 'quantization'
    fc1w, quantized_fc1w, fc1w_scale = quantization('mnist_dkdk_FP32_20170707_v3/FC1.npy')
    fc2w, quantized_fc2w, fc2w_scale = quantization('mnist_dkdk_FP32_20170707_v3/FC2.npy')
    fc3w, quantized_fc3w, fc3w_scale = quantization('mnist_dkdk_FP32_20170707_v3/FC3.npy')
    if run_mode == 'quantization':
        fc1w = quantized_fc1w
        fc2w = quantized_fc2w
        fc3w = quantized_fc3w
        pass

    temp = np.matmul(inp, fc1w)
    temp = temp * fc1w_scale
    temp = np.maximum(0, temp)

    temp = np.matmul(temp, fc2w)
    temp = temp * fc2w_scale
    temp = np.maximum(0, temp)
    
    temp = np.matmul(temp, fc3w)
    temp = temp * fc3w_scale
    # temp = temp * fc1w_scale * fc2w_scale * fc3w_scale

    print(temp)

    from scipy.special import softmax
    temp = softmax(temp)
    result = np.argmax(temp)
    print(temp)
    print(result)

    return result


if __name__ == '__main__':

    ret = main('test_image/4.png')
    print(ret)

