
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
 

def print_debug_hex(input_array):
    """
    Print HEX

    :param np.ndarray array: input array
    :return: None
    """
    array = input_array.view(dtype=np.uint8)

    terminal_rows, terminal_columns = map(int, os.popen('stty size', 'r').read().split())

    print_hex_rows = min(array.shape[0] - 1, terminal_rows - 5)
    print_hex_columns = min(array.shape[1] - 1, (terminal_columns - 16) // 3)

    if input_array.dtype == np.int8 or input_array.dtype == np.uint8:
        print_hex_columns -= (print_hex_columns + 1) % 2
    elif input_array.dtype == np.int16 or input_array.dtype == np.uint16:
        print_hex_columns -= (print_hex_columns + 1) % 4

    rows_omitted = array.shape[0] - 1 > terminal_rows - 5
    columns_omitted = array.shape[1] - 1 > (terminal_columns - 16) // 3

    if __debug__:
        print('print_hex_rows :', print_hex_rows)
        print('print_hex_columns :', print_hex_columns)
        print('rows_omitted :', rows_omitted)
        print('columns_omitted :', columns_omitted)

    msgs = []
    # ..........0.........1....
    # ..........01234567890123
    msgs.append('        dd dddd ') # 0
    msgs.append('      ┌────┐') # 1
    msgs.append('   dd │ xx │') # 2
    msgs.append(' dddd └────┘') # 3

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
        # dddd -> {:4} 
        msgs[i] = msgs[i].replace('dddd', colored('{:4}', 'green'))

        # xx -> {:02X}
        if input_array.dtype == np.int8 or input_array.dtype == np.uint8:
            temp = colored('{:02X} ', 'green') + colored('{:02X}', 'red')
            msgs[i] = msgs[i].replace('xx xx', temp)
        elif input_array.dtype == np.int16 or input_array.dtype == np.uint16:
            temp = colored('{:02X} {:02X} ', 'green') + colored('{:02X} {:02X}', 'red')
            msgs[i] = msgs[i].replace('xx xx xx xx', temp)

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

    # array = np.random.randint(0xFF, size=(500, 3000), dtype=np.uint8)
    array = np.random.randint(0xFFFF, size=(500, 3000), dtype=np.uint16)
    print(array)
    print_debug_hex(array)
    pass

if __name__ == '__main__':
    test()

