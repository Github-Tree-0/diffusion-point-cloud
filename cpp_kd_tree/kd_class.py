import ctypes
from tokenize import Double
import numpy as np
import os

# path = './kd.so'
lib = ctypes.cdll.LoadLibrary("./kd.so")

class KDTree():
    def __init__(self, points):
        # points = [num_points, 3]
        points_ptr = points.ctypes.data_as(ctypes.c_char_p)
        self.pts_num = points.shape[0]
        self.obj = lib.kd_new(points_ptr, self.pts_num)
    
    def build(self, at, l, r, d):
        lib.build(self.obj, at, l, r, d)

    def output(self):
        index = np.zeros(self.pts_num)
        index_ptr = index.ctypes.data_as(ctypes.c_char_p)
        lib.output_index(self.obj, index_ptr)

        return index
