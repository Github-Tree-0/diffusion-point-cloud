# distutils: language = c++
from kd_class_d cimport KDTree
import numpy as np
cimport numpy as np
import ctypes
import sys

cdef class PyKDTree:
    cdef KDTree *c_tree
    cdef int num_pts
    def __cinit__(self, np.ndarray[np.double_t, ndim=2] points):
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] input_arr
        input_arr = np.ascontiguousarray(points, dtype=np.double)
        self.num_pts = points.shape[0]
        self.c_tree = new KDTree(<double *> input_arr.data, self.num_pts)
    def build(self, int at, int l, int r, int d):
        self.c_tree.build(at, l, r, d)
    def output_index(self):
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] output_arr
        output_arr = np.ascontiguousarray(np.zeros(self.num_pts), dtype=np.int32)
        self.c_tree.output_index(<int *> output_arr.data)
        return output_arr
    def __dealloc__(self):
        del self.c_tree