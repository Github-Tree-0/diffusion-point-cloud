cdef extern from "kd_class_src.cpp":
    pass
cdef extern from "kd_class.h":
    const int MAXN
    struct Point:
        pass
    struct Node:
        pass
    int dim

    cdef cppclass KDTree:
        int num
        Point p[MAXN]
        Node Tree[MAXN << 2]
        KDTree() except +
        KDTree(double *input_arr, int num_pts) except +
        void build(int at, int l, int r, int d)
        void output_index(int *output_arr)