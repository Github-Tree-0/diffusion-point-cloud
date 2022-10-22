cdef extern from "kd_def.cpp":
    pass
cdef extern from "kd_def.h":
    const int MAXN
    struct Point:
        pass
    int dim

    cdef cppclass KDTree:
        int num
        Point p1[MAXN], p2[MAXN]
        KDTree() except +
        KDTree(double *points_1, double *points_2, int num_pts) except +
        void build(int at, int l, int r, int d)
        void output_index(int *out_index1, int *out_index2)