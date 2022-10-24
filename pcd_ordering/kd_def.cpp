#include "kd_def.h"
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

KDTree::KDTree(double *points_1, double *points_2, int32_t num_pts) {
    this->num = num_pts;
    int32_t ptr = 0;
    for (int i = 1; i <= num_pts; ++i) {
        this->p1[i].index = i - 1;
        this->p2[i].index = i - 1;
        for (int j = 0; j < 3; ++j) {
            this->p1[i].cor[j] = double(points_1[ptr]);
            this->p2[i].cor[j] = double(points_2[ptr++]);
        }
    }
}

void KDTree::build(int32_t at, int32_t l, int32_t r, int32_t d) {
    this->rotate_points(l, r);
    int32_t m = (l + r) >> 1;
    dim = d;
    std::nth_element(this->p1 + l, this->p1 + m, this->p1 + r + 1, cmp);
    std::nth_element(this->p2 + l, this->p2 + m, this->p2 + r + 1, cmp);
    double max_1[3], min_1[3], max_2[3], min_2[3];
    for (int i = 0; i < 3; ++i) {
        max_1[i] = -1e100;
        max_2[i] = -1e100;
        min_1[i] = 1e100;
        min_2[i] = 1e100;
    }
    for (int i = l; i <= r; ++i) {
        for (int j = 0; j < 3; ++j) {
            cmin(min_1[j], this->p1[i].cor[j]);
            cmin(min_2[j], this->p2[i].cor[j]);
            cmax(max_1[j], this->p1[i].cor[j]);
            cmax(max_2[j], this->p2[i].cor[j]);
        }
    }
    double max_min_var = -1.0;
    for (int i = 0; i < 3; ++i){
        double var1 = max_1[i] - min_1[i];
        double var2 = max_2[i] - min_2[i];
        double min_var = (var1 < var2 ? var1 : var2);
        if (min_var > max_min_var) {
            max_min_var = min_var;
            d = i;
        }
    }
    if(l < r) {
        build(at << 1, l, m, d);
        build(at << 1 | 1, m + 1, r, d);
    }
}

void KDTree::output_index(int *out_index1, int *out_index2) {
    for (int i = 1; i <= this->num; ++i) {
        out_index1[i-1] = this->p1[i].index;
        out_index2[i-1] = this->p2[i].index;
    }
}

void KDTree::rotate_points(int32_t l, int32_t r) {
    double X = rand_angle(), Y = rand_angle(), Z = rand_angle();
    double c1 = cos(Y), c2 = cos(X), c3 = cos(Z);
    double s1 = sin(Y), s2 = sin(X), s3 = sin(Z);
    for (int i = l; i <= r; ++i) {
        double x, y, z;
        // Rotate p1
        x = this->p1[i].cor[0]; y = this->p1[i].cor[1]; z = this->p1[i].cor[2];
        this->p1[i].cor[0] = x*(c1*c3+s1*s2*s3) + y*(c3*s1*s2-c1*s3) + z*(c2*s1);
        this->p1[i].cor[1] = x*(c2*s3) + y*(c2*s3) - z*(s2);
        this->p1[i].cor[2] = x*(c1*s2*s3-s1*c3) + y*(s1*s3+c1*c3*s2) + z*(c1*c2);
        // Rotate p2
        x = this->p2[i].cor[0]; y = this->p2[i].cor[1]; z = this->p2[i].cor[2];
        this->p2[i].cor[0] = x*(c1*c3+s1*s2*s3) + y*(c3*s1*s2-c1*s3) + z*(c2*s1);
        this->p2[i].cor[1] = x*(c2*s3) + y*(c2*s3) - z*(s2);
        this->p2[i].cor[2] = x*(c1*s2*s3-s1*c3) + y*(s1*s3+c1*c3*s2) + z*(c1*c2);
    }
}