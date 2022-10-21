#pragma once

#define For(i, a, b) for(int i = a, ___u = b; i <= ___u; ++i)
#define ForDown(i, a, b) for(int i = b, ___d = a; i >= ___d; --i)
#define cmax(i, j) ((i) < (j) ? (i) = (j) : (i))
#define cmin(i, j) ((i) > (j) ? (i) = (j) : (i))
const int MAXN = 33554432;

struct Point {double cor[3]; int index;};
struct Node {double min[3], max[3]; int size;};
int dim;
inline bool cmp(const Point& a, const Point& b) { return a.cor[dim] < b.cor[dim]; }

class KDTree {
public:
    int num;
    Point p[MAXN];
    Node Tree[MAXN << 2];

    KDTree();
    KDTree(double *input_arr, int num_pts);

    void build(int at, int l, int r, int d);

    void output_index(int *output_arr);
};