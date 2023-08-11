#include <iostream>
#include <stdint.h>

#include "iob_unum4.h"

class Unum4 {
    private:
        unum4 val = 0;
        uint8_t failed = 0, overflow = 0, underflow = 0, div_by_zero = 0;
    public:
        Unum4(): val() {};
        Unum4(float f): val(float2unum4(f, &failed)) {};
        Unum4(double d): val(double2unum4(d, &failed)) {};
        Unum4(int i): val(double2unum4((double)i, &failed)) {};

        Unum4 operator+(Unum4 const& other) const {
            Unum4 res;
            res.val = unum4_add(val, other.val, &res.overflow);
            return res;
        }

        Unum4 operator-(Unum4 const& other) const {
            Unum4 res;
            res.val = unum4_sub(val, other.val, &res.overflow);
            return res;
        }

        Unum4 operator*(Unum4 const& other) const {
            Unum4 res;
            res.val = unum4_mul(val, other.val, &res.overflow, &res.underflow);
            return res;
        }

        Unum4 operator/(Unum4 const& other) const {
            Unum4 res;
            res.val = unum4_div(val, other.val, &res.overflow, &res.underflow, &res.div_by_zero);
            return res;
        }
        // conversion operator could be handy
        operator float() { return unum42float(val); }
        operator double() { return unum42double(val); }
        operator int() { return (int)unum42double(val); }
};

int main(int argc, char **argv) {
    Unum4 a = 1.0;
    Unum4 b = 2.0;
    Unum4 c = a + b;
    std::cout << (double)c << std::endl;
    c = a - b;
    std::cout << (double)c << std::endl;
    c = a * b;
    std::cout << (double)c << std::endl;
    c = a / b;
    std::cout << (double)c << std::endl;
    return 0;
}
