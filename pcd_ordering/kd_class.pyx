# distutils: language = c++
from kd_class cimport KDTree
import numpy as np
cimport numpy as np
import ctypes
import sys

cdef class PyKDTree:
    cdef KDTree *c_tree
    cdef int num_pts

    def __cinit__(self, np.ndarray[np.double_t, ndim=2] points_1, np.ndarray[np.double_t, ndim=2] points_2):
        assert(points_1.shape[0] == points_2.shape[0])
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] input_arr1
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] input_arr2
        input_arr1 = np.ascontiguousarray(points_1, dtype=np.double)
        input_arr2 = np.ascontiguousarray(points_2, dtype=np.double)
        self.num_pts = points_1.shape[0]
        self.c_tree = new KDTree(<double *> input_arr1.data, <double *>input_arr2.data, self.num_pts)

    def build(self, int at, int l, int r, int d):
        self.c_tree.build(at, l, r, d)

    def output_index(self):
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] output_arr1
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] output_arr2
        output_arr1 = np.ascontiguousarray(np.zeros(self.num_pts), dtype=np.int32)
        output_arr2 = np.ascontiguousarray(np.zeros(self.num_pts), dtype=np.int32)
        self.c_tree.output_index(<int *> output_arr1.data, <int *> output_arr2.data)
        return output_arr1, output_arr2

    def __dealloc__(self):
        del self.c_tree