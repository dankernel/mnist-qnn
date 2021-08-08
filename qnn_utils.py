
import os
import numpy as np
from termcolor import colored

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
        print('dim :', dim)
        file.write(dim.to_bytes(4, byteorder='little', signed=True))
        for s in range(dim):
            size = ndarray.shape[s]
            print('size :', size)
            file.write(size.to_bytes(4, byteorder='little', signed=True))
        file.write(ndarray.tobytes())
 

def print_debug_hex(array):
    """
    Print HEX

    :param np.ndarray array: input array
    :return: None
    """

    terminal_rows, terminal_columns = map(int, os.popen('stty size', 'r').read().split())

    print_hex_rows = min(array.shape[0] - 1, terminal_rows - 5)
    print_hex_columns = min(array.shape[1] - 1, (terminal_columns - 16) // 3)
    rows_omitted = array.shape[0] - 1 > terminal_rows - 5
    columns_omitted = array.shape[1] - 1 > (terminal_columns - 16) // 3

    msgs = []
    # ..........0.........1....
    # ..........01234567890123
    msgs.append('        dd xxxx ') # 0
    msgs.append('      ┌────┐') # 1
    msgs.append('   dd │ xx │') # 2
    msgs.append(' xxxx └────┘') # 3

    # columns(X-axis) extend
    for i in range(len(msgs)):
        for j in range(print_hex_columns):
            if i == 0:
                msgs[i] = msgs[i][:7] + ' dd' + msgs[i][7:]
            elif i == 1 or i == 3:
                msgs[i] = msgs[i][:7] + '───' + msgs[i][7:]
            else:
                msgs[i] = msgs[i][:7] + ' xx' + msgs[i][7:]

    # rows(Y-axis) extend
    for i in range(print_hex_rows):
        msgs.insert(2, msgs[2])

    for i in range(len(msgs)):
        msgs[i] = msgs[i].replace('xxxx', colored('{:4}', 'green'))
        msgs[i] = msgs[i].replace('xx', '{:02x}')
        msgs[i] = msgs[i].replace('dd', '{:02}')

    # print all
    for i in range(len(msgs)):
        if i == 0:
            temp = list(range(print_hex_columns + 1))
            tepm = temp.append(array.shape[1])
            print(msgs[i].format(*temp))
        elif i == len(msgs) - 1:
            print(msgs[i].format(array.shape[0]))
        else:
            temp = list(array[i-2])
            tepm = temp.insert(0, i - 2)
            print(msgs[i].format(*temp))

    return

def test():

    array = np.random.randint(10, size=(5, 3000), dtype=np.uint8)
    print(array)
    print_debug_hex(array)
    pass

if __name__ == '__main__':
    test()

