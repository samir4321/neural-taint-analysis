# utils.py
import numpy as np
import torch
from torch import __init__

def str_to_bytes(s):
    # utf-8 encode string
    return s.encode("utf-8")

def byte_to_num(b):
    # convert bytes to int
    return int.from_bytes(b, "big", signed=True)

def num_to_byte(n):
    # int to bytes
    return n.to_bytes((n.bit_length() + 7) // 8, 'big', signed=True) or b'\0'

def normalize(byte_int):
    # ord in 0 to 255 mapped into [-1, 1]
    return (byte_int - 128) / 256.

def denormalize(xn):
    # xn in (-1, 1) into 0 to 255
    # must be exact inverse of normalize
    return int((xn * 256) + 128)

def matprint(mat, fmt="g"):
    # pretty print a numpy array as matrix
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col
                 in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")

def map_tensor(x, fn):
    x_arr = x.cpu().detach().numpy()
    z_arr = np.array([fn(e) for e in x_arr])
    z = torch.FloatTensor(z_arr).reshape(z_arr.shape)
    return z

