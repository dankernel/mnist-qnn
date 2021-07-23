
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
 
