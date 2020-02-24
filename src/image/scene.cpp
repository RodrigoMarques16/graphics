#include "scene.hpp"

void Scene::addTriangle(const Triangle& t) {
    triangles.emplace_back(t);
}

void Scene::addTriangle(const Vec3& p0, const Vec3& p1, const Vec3& p2, const rgb& color) {
    triangles.emplace_back(p0, p1, p2, color);
}