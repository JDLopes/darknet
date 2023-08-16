#include <iostream>
#include <stdint.h>

#include "iob_unum4.h"

uint8_t failed, overflow, underflow, div_by_zero;

class Unum4 {
    private:
        unum4 val = 0;
    public:
        Unum4(): val() {};
        Unum4(float f): val(float2unum4(f, &failed)) {};
        Unum4(double d): val(double2unum4(d, &failed)) {};
        Unum4(int i): val(double2unum4((double)i, &failed)) {};

        // Arithmetic operators
        Unum4 operator+(Unum4 const& other) const {
            Unum4 res;
            res.val = unum4_add(val, other.val, &overflow);
            return res;
        }

        Unum4 operator+=(Unum4 const& other) {
            return val = unum4_add(val, other.val, &overflow);
        }

        Unum4 operator-(Unum4 const& other) const {
            Unum4 res;
            res.val = unum4_sub(val, other.val, &overflow);
            return res;
        }

        Unum4 operator-=(Unum4 const& other) {
            return val = unum4_sub(val, other.val, &overflow);
        }

        Unum4 operator*(Unum4 const& other) const {
            Unum4 res;
            res.val = unum4_mul(val, other.val, &overflow, &underflow);
            return res;
        }

        Unum4 operator*=(Unum4 const& other) {
            return val = unum4_mul(val, other.val, &overflow, &underflow);
        }

        Unum4 operator/(Unum4 const& other) const {
            Unum4 res;
            res.val = unum4_div(val, other.val, &overflow, &underflow, &div_by_zero);
            return res;
        }

        Unum4 operator/=(Unum4 const& other) {
            return val = unum4_div(val, other.val, &overflow, &underflow, &div_by_zero);
        }

        // Comparator operators
        bool operator<(Unum4 const& other) { return (bool)unum4_lt(val, other.val); }
        bool operator<=(Unum4 const& other) { return (bool)unum4_le(val, other.val); }
        bool operator>(Unum4 const& other) { return (bool)unum4_gt(val, other.val); }
        bool operator>=(Unum4 const& other) { return (bool)unum4_ge(val, other.val); }
        bool operator==(Unum4 const& other) { return (bool)unum4_eq(val, other.val); }
        bool operator!=(Unum4 const& other) { return (bool)!unum4_eq(val, other.val); }

        // Conversion operators
        operator float() { return unum42float(val); }
        operator double() { return unum42double(val); }
        operator int() { return (int)unum42double(val); }
};

int main(int argc, char **argv) {
    Unum4 a = 1.0;
    Unum4 b = 2.0;

    // Arithmetics
    Unum4 c = a + b;
    std::cout << (double)c << std::endl;
    c += b;
    std::cout << (double)c << std::endl;
    c = a - b;
    std::cout << (double)c << std::endl;
    c -= b;
    std::cout << (double)c << std::endl;
    c = a * b;
    std::cout << (double)c << std::endl;
    c *= b;
    std::cout << (double)c << std::endl;
    c = a / b;
    std::cout << (double)c << std::endl;
    c /= b;
    std::cout << (double)c << std::endl;

    // Comparators
    if (a > b) {
      std::cout << "a > b" << std::endl;
    } else {
      std::cout << "a !> b" << std::endl;
    }
    if (a >= b) {
      std::cout << "a >= b" << std::endl;
    } else {
      std::cout << "a !>= b" << std::endl;
    }
    if (a < b) {
      std::cout << "a < b" << std::endl;
    } else {
      std::cout << "a !< b" << std::endl;
    }
    if (a <= b) {
      std::cout << "a <= b" << std::endl;
    } else {
      std::cout << "a !<= b" << std::endl;
    }
    if (a == b) {
      std::cout << "a == b" << std::endl;
    } else {
      std::cout << "a !== b" << std::endl;
    }
    if (a != b) {
      std::cout << "a != b" << std::endl;
    } else {
      std::cout << "a !!= b" << std::endl;
    }

    // Conversions
    c = 37.48;
    std::cout << (double)c << std::endl;
    std::cout << (float)c << std::endl;
    std::cout << (int)c << std::endl;

    return 0;
}
