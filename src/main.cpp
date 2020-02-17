#include "math/vec3.hpp"
#include "math/rgb.hpp"
#include "math/unit3.hpp"
#include "math/onb.hpp"

int main() {
    static_assert(std::is_same<std::is_same<int, char>::value_type, bool>::value);
}