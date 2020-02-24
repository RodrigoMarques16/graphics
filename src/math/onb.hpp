#pragma once

#include <ostream>
#include "unit3.hpp"

struct ONB {
    static constexpr const float eps = 0.01f;

    Unit3 u, v, w;

    static ONB fromU(const Unit3& xx) {
        auto yy = xx.cross(Unit3::xAxis);
        if (yy.lengthSquared() < eps)
            yy = xx.cross(Unit3::yAxis);
        auto zz = xx.cross(yy);
        return { xx, yy.unitVector(), zz.unitVector() };
    }

    static ONB fromV(const Unit3& yy) {
        auto xx = yy.cross(Unit3::xAxis);
        if (xx.lengthSquared() < eps)
            xx = yy.cross(Unit3::yAxis);
        auto zz = xx.cross(yy);
        return {xx.unitVector(), yy, zz.unitVector()};
    }

    static ONB fromW(const Unit3& zz) {
        auto xx = zz.cross(Unit3::xAxis);
        if (xx.lengthSquared() < eps)
            xx = zz.cross(Unit3::yAxis);
        auto yy = xx.cross(zz);
        return {xx.unitVector(), yy.unitVector(), zz};
    }

    static ONB fromUV(const Unit3& xx, const Unit3& u) {
        auto zz = xx.cross(u).unitVector();
        auto yy = zz.cross(xx).unitVector();
        return {xx, yy, zz};
    }

    static ONB fromVU(const Unit3& yy, const Unit3& u) {
        auto zz = u.cross(yy).unitVector();
        auto xx = yy.cross(zz).unitVector();
        return {xx, yy, zz};
    }

    static ONB fromUW(const Unit3& xx, const Unit3& w) {
        auto yy = w.cross(xx).unitVector();
        auto zz = xx.cross(yy).unitVector();
        return {xx, yy, zz};
    }
    
    static ONB fromWU(const Unit3& zz, const Unit3& u) {
        auto yy = zz.cross(u).unitVector();
        auto xx = yy.cross(zz).unitVector();
        return {xx, yy, zz};
    }

    static ONB fromVW(const Unit3& yy, const Unit3& w) {
        auto xx = yy.cross(w).unitVector();
        auto zz = xx.cross(yy).unitVector();
        return {xx, yy, zz};
    }

    static ONB fromWV(const Unit3& zz, const Unit3& v) {
        auto xx = v.cross(zz).unitVector();
        auto yy = zz.cross(xx).unitVector();
        return {xx, yy, zz};
    }

    bool operator==(const ONB& other) {
        return u == other.u
            && v == other.v
            && w == other.w;
    }

};

std::ostream& operator<<(std::ostream& os, const ONB& o);