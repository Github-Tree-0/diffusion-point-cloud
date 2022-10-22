#include <cstdint>
#pragma once

#define cmax(i, j) ((i) < (j) ? (i) = (j) : (i))
#define cmin(i, j) ((i) > (j) ? (i) = (j) : (i))
const int MAXN = 33554432;

struct Point {double cor[3]; int index;};

int dim;
inline bool cmp(const Point& a, const Point& b) { return a.cor[dim] < b.cor[dim]; }

class KDTree {
public:
    int num;
    Point p1[MAXN], p2[MAXN];

    KDTree();
    KDTree(double *points_1, double *points_2, int32_t num_pts);

    void build(int at, int l, int r, int d);
    void output_index(int *out_index1, int *out_index2);
};