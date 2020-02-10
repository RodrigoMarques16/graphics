#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <vec3.hpp>

TEST_CASE("Vectors", "[math]") {
    SECTION("default construction") {
        auto vec = Vec3();
        CHECK(vec.x == 0);
        CHECK(vec.y == 0);
        CHECK(vec.z == 0);
    }

    SECTION("construction") {
        auto vec = Vec3(1, 2, 3);
        CHECK(vec.x == 1);
        CHECK(vec.y == 2);
        CHECK(vec.z == 3);
    }

    SECTION("comparison") {
        CHECK(Vec3() == Vec3());
        CHECK(Vec3(1, 2, 3) == Vec3(1, 2, 3));
        CHECK(Vec3(1, 2, 3) == Vec3(1.00000001, 2.00000001, 3.00000001));

        CHECK(Vec3() != Vec3(1, 2, 3));
        CHECK(Vec3(1, 2, 3) != Vec3());
        CHECK(Vec3(1, 2, 3) != Vec3(3, 2, 1));
    }

    SECTION("negation") {
        CHECK(-Vec3() == Vec3());
        CHECK(-Vec3(1, 2, 3) == Vec3(-1, -2, -3));
    }

    SECTION("addition") {
        CHECK(Vec3() + Vec3() == Vec3());
        CHECK(Vec3(1, 2, 3) + Vec3(1, 2, 3) == Vec3(2, 4, 6));

        auto vec = Vec3(1, 2, 3);
        vec += vec;

        CHECK(vec == Vec3(2, 4, 6));
    }

    SECTION("subtraction") {
        CHECK(Vec3() - Vec3() == Vec3());
        CHECK(Vec3(1, 2, 3) - Vec3(1, 2, 3) == Vec3());
        CHECK(Vec3(1, 2, 3) - Vec3(1, 1, 1) == Vec3(0, 1, 2));

        auto vec = Vec3(1, 2, 3);
        vec -= vec;

        CHECK(vec == Vec3());
    }

    SECTION("scalar") {
        CHECK(Vec3() * 0 == Vec3());
        CHECK(0 * Vec3() == Vec3());

        CHECK(Vec3(1, 2, 3) * 0 == Vec3());
        CHECK(Vec3(1, 2, 3) * 1 == Vec3(1, 2, 3));
        CHECK(Vec3(1, 2, 3) * 2 == Vec3(2, 4, 6));

        auto vec = Vec3(1, 2, 3);
        vec *= 2;

        CHECK(vec == Vec3(2, 4, 6));
    }

    SECTION("scalar division") {
        CHECK(Vec3() / 1 == Vec3());

        CHECK(Vec3(1, 2, 3) / 1 == Vec3(1, 2, 3));
        CHECK(Vec3(2, 4, 6) / 2 == Vec3(1, 2, 3));

        auto vec = Vec3(2, 4, 6);
        vec /= 2;

        CHECK(vec == Vec3(1, 2, 3));
    }

    SECTION("multiplication") {
        CHECK(Vec3() * Vec3() == Vec3());
        CHECK(Vec3(1, 2, 3) * Vec3(1, 2, 3) == Vec3(1, 4, 9));

        auto vec = Vec3(1, 2, 3);
        vec *= Vec3(2, 2, 2);

        CHECK(vec == Vec3(2, 4, 6));
    }

    SECTION("dot product") {
        CHECK(Vec3().dot(Vec3()) == 0);
        CHECK(Vec3().dot(Vec3(1, 2, 3)) == 0);
        CHECK(Vec3(1, 2, 3).dot(Vec3()) == 0);
        CHECK(Vec3(1, 2, 3).dot(Vec3(1, 2, 3)) == 14);

        CHECK(Vec3(1, 2, 3).dot(Vec3::xAxis) == 1);
        CHECK(Vec3(1, 2, 3).dot(Vec3::yAxis) == 2);
        CHECK(Vec3(1, 2, 3).dot(Vec3::zAxis) == 3);
    }

    SECTION("cross product") {
        CHECK(Vec3().cross(Vec3()) == Vec3());
        CHECK(Vec3().cross(Vec3(1, 2, 3)) == Vec3());
        CHECK(Vec3(1, 2, 3).cross(Vec3()) == Vec3());
        CHECK(Vec3(1, 2, 3).cross(Vec3(1, 2, 3)) == Vec3());
        CHECK(Vec3(1, 2, 3).cross(Vec3(3, 2, 1)) == Vec3(-4, 8, -4));
        CHECK(Vec3::xAxis.cross(Vec3::yAxis) == Vec3::forward);
        CHECK(Vec3::yAxis.cross(Vec3::xAxis) == Vec3::back);
    }

    SECTION("magnitude") {
        CHECK(Vec3().lengthSquared() == 0);
        CHECK(Vec3().length() == 0);
        CHECK(Vec3(1, 2, 3).lengthSquared() == 14);
        CHECK(Vec3(1, 2, 3).length() == Approx(sqrt(14)));
    }

    SECTION("normalize") {
        CHECK(Vec3().normalised() == Vec3::zero);
        CHECK(Vec3(2, 0, 0).normalised() == Vec3::xAxis);
        CHECK(Vec3(1, 2, 3).normalised().length() == Approx(1.0));
    }
}