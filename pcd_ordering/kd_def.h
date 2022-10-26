#pragma once
#include <cstdint>
#include <cstdlib>
#include <time.h>

#define cmax(i, j) ((i) < (j) ? (i) = (j) : (i))
#define cmin(i, j) ((i) > (j) ? (i) = (j) : (i))
const int MAXN = 33554432;
const double Pi_2 = 2 * 3.141592653589793115997963468;

struct Point {double cor[3]; int index;};

int dim;
inline bool cmp(const Point& a, const Point& b) { return a.cor[dim] < b.cor[dim]; }
inline double rand_angle() {
    return std::rand() / double(RAND_MAX) * Pi_2;
}

class KDTree {
public:
    int num;
    Point p1[MAXN], p2[MAXN];

    KDTree();
    KDTree(double *points_1, double *points_2, int32_t num_pts);

    void build(int32_t at, int32_t l, int32_t r, int32_t d);
    void output_index(int *out_index1, int *out_index2);
    void rotate_points(int32_t l, int32_t r);
};