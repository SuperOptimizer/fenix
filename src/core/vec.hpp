// core/vec.hpp — small first-party linear algebra (no Eigen). Spatial vectors are
// axis-tagged (z,y,x) to kill the normal-axis-order bug class (docs/conventions.md).
#pragma once

#include "core/types.hpp"

#include <cmath>

namespace fenix {

// Axis-tagged 3-vector in ZYX order. Use for positions, directions, normals.
template <class T>
struct Vec3 {
    T z{}, y{}, x{};

    constexpr Vec3() = default;
    constexpr Vec3(T z_, T y_, T x_) : z(z_), y(y_), x(x_) {}

    friend constexpr Vec3 operator+(Vec3 a, Vec3 b) { return {a.z + b.z, a.y + b.y, a.x + b.x}; }
    friend constexpr Vec3 operator-(Vec3 a, Vec3 b) { return {a.z - b.z, a.y - b.y, a.x - b.x}; }
    friend constexpr Vec3 operator*(Vec3 a, T s) { return {a.z * s, a.y * s, a.x * s}; }
    friend constexpr Vec3 operator*(T s, Vec3 a) { return a * s; }
    friend constexpr Vec3 operator/(Vec3 a, T s) { return {a.z / s, a.y / s, a.x / s}; }
    friend constexpr bool operator==(Vec3, Vec3) = default;
};

using Vec3f = Vec3<f32>;
using Vec3d = Vec3<f64>;

template <class T>
constexpr T dot(Vec3<T> a, Vec3<T> b) {
    return a.z * b.z + a.y * b.y + a.x * b.x;
}

template <class T>
constexpr Vec3<T> cross(Vec3<T> a, Vec3<T> b) {
    // Right-handed a×b, components stored ZYX: c_z = a_x b_y − a_y b_x, etc.
    // (x̂×ŷ=ẑ: cross({0,0,1},{0,1,0}) == {1,0,0}.)
    return {a.x * b.y - a.y * b.x, a.z * b.x - a.x * b.z, a.y * b.z - a.z * b.y};
}

template <class T>
T norm(Vec3<T> a) {
    return std::sqrt(dot(a, a));
}

template <class T>
Vec3<T> normalized(Vec3<T> a) {
    const T n = norm(a);
    return n > T(0) ? a / n : a;
}

// Row-major 3x3 matrix (rows/cols in ZYX). Enough for structure-tensor / affine work;
// a symmetric-3x3 eigensolver and SVD land alongside the segment front end.
template <class T>
struct Mat3 {
    T m[3][3]{};

    static constexpr Mat3 identity() {
        Mat3 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = T(1);
        return r;
    }

    constexpr Vec3<T> operator*(Vec3<T> v) const {
        return {m[0][0] * v.z + m[0][1] * v.y + m[0][2] * v.x,
                m[1][0] * v.z + m[1][1] * v.y + m[1][2] * v.x,
                m[2][0] * v.z + m[2][1] * v.y + m[2][2] * v.x};
    }

    constexpr Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j];
        return r;
    }
};

using Mat3f = Mat3<f32>;
using Mat3d = Mat3<f64>;

}  // namespace fenix
