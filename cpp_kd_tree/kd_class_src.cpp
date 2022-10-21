#include "kd_class.h"
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>
#include <set>
#include <utility>
#include <queue>
#include <cstdint>

KDTree::KDTree() {}

KDTree::KDTree(double *input_arr, int32_t num_pts) {
    this->num = num_pts;
    int32_t ptr = 0;
    For(i, 1, num_pts) {
        this->p[i].index = i-1;
        For(j, 0, 2)
            this->p[i].cor[j] = double(input_arr[ptr++]);
    }
}

void KDTree::build(int32_t at, int32_t l, int32_t r, int32_t d) {
    int32_t m = (l + r) >> 1;
    dim = d;
    std::nth_element(this->p + l, this->p + m, this->p + r + 1, cmp);
    this->Tree[at].size = r - l + 1;
    For(i, 0, 2) {
        this->Tree[at].min[i] = 1e100;
        this->Tree[at].max[i] = -1e100;
    }
    For(i, l, r) For(j, 0, 2) {
        cmin(this->Tree[at].min[j], this->p[i].cor[j]);
        cmax(this->Tree[at].max[j], this->p[i].cor[j]);
    }
    d-- ? 1 : d += 3;
    if(l < r) {
        build(at << 1, l, m, d);
        build(at << 1 | 1, m + 1, r, d);
    }
}

void KDTree::output_index(int32_t *output_arr) {
    For(i, 1, this->num)
        output_arr[i-1] = this->p[i].index;
}