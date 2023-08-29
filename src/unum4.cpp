#include <iostream>
#include <stdint.h>

#include "iob_unum4.h"

uint8_t failed, overflow, underflow, div_by_zero;

class Unum4 {
    private:
        unum4 val;
    public:

        //
        // Declarations
        //

        Unum4(): val(float2unum4(0.0, &failed)) {};
        Unum4(float f): val(float2unum4(f, &failed)) {};
        Unum4(double d): val(double2unum4(d, &failed)) {};
        Unum4(int i): val(double2unum4((double)i, &failed)) {};

        //
        // Conversion operators
        //

        operator float() { return unum42float(val); }
        operator double() { return unum42double(val); }
        operator int() { return (int)unum42double(val); }

        //
        // Arithmetic operators
        //

        // Addition
        Unum4 operator++() {
            Unum4 res = 1.0;
            res.val = unum4_add(val, res.val, &overflow);
            return res;
        }

        Unum4 operator++( int ) {
            Unum4 res;
            res.val = val;
            val = unum4_add(val, ((Unum4)1.0).val, &overflow);
            return res;
        }

        Unum4 operator+(Unum4 const& other) {
            Unum4 res;
            res.val = unum4_add(val, other.val, &overflow);
            return res;
        }

        Unum4 operator+(double const other) {
            Unum4 res = other;
            res.val = unum4_add(val, res.val, &overflow);
            return res;
        }

        Unum4 operator+(float const other) {
            Unum4 res = other;
            res.val = unum4_add(val, res.val, &overflow);
            return res;
        }

        Unum4 operator+(int const other) {
            Unum4 res = other;
            res.val = unum4_add(val, res.val, &overflow);
            return res;
        }

        Unum4 operator+=(Unum4 const& other) {
            return val = unum4_add(val, other.val, &overflow);
        }

        // Subtraction
        Unum4 operator--() {
            Unum4 res = 1.0;
            res.val = unum4_sub(val, res.val, &overflow);
            return res;
        }

        Unum4 operator--( int ) {
            Unum4 res;
            res.val = val;
            val = unum4_sub(val, ((Unum4)1.0).val, &overflow);
            return res;
        }

        Unum4 operator-() {
            Unum4 res = 0.0;
            res.val = unum4_sub(res.val, val, &overflow);
            return res;
        }

        Unum4 operator-(Unum4 const& other) {
            Unum4 res;
            res.val = unum4_sub(val, other.val, &overflow);
            return res;
        }

        Unum4 operator-(double const other) {
            Unum4 res = other;
            res.val = unum4_sub(val, res.val, &overflow);
            return res;
        }

        Unum4 operator-(float const other) {
            Unum4 res = other;
            res.val = unum4_sub(val, res.val, &overflow);
            return res;
        }

        Unum4 operator-(int const other) {
            Unum4 res = other;
            res.val = unum4_sub(val, res.val, &overflow);
            return res;
        }

        Unum4 operator-=(Unum4 const& other) {
            return val = unum4_sub(val, other.val, &overflow);
        }

        // Multiplication
        Unum4 operator*(Unum4 const& other) {
            Unum4 res;
            res.val = unum4_mul(val, other.val, &overflow, &underflow);
            return res;
        }

        Unum4 operator*(double const other) {
            Unum4 res = other;
            res.val = unum4_mul(val, res.val, &overflow, &underflow);
            return res;
        }

        Unum4 operator*(float const other) {
            Unum4 res = other;
            res.val = unum4_mul(val, res.val, &overflow, &underflow);
            return res;
        }

        Unum4 operator*(int const other) {
            Unum4 res = other;
            res.val = unum4_mul(val, res.val, &overflow, &underflow);
            return res;
        }

        Unum4 operator*=(Unum4 const& other) {
            return val = unum4_mul(val, other.val, &overflow, &underflow);
        }

        // Division
        Unum4 operator/(Unum4 const& other) {
            Unum4 res;
            res.val = unum4_div(val, other.val, &overflow, &underflow, &div_by_zero);
            return res;
        }

        Unum4 operator/(double const other) {
            Unum4 res = other;
            res.val = unum4_div(val, res.val, &overflow, &underflow, &div_by_zero);
            return res;
        }

        Unum4 operator/(float const other) {
            Unum4 res = other;
            res.val = unum4_div(val, res.val, &overflow, &underflow, &div_by_zero);
            return res;
        }

        Unum4 operator/(int const other) {
            Unum4 res = other;
            res.val = unum4_div(val, res.val, &overflow, &underflow, &div_by_zero);
            return res;
        }

        Unum4 operator/=(Unum4 const& other) {
            return val = unum4_div(val, other.val, &overflow, &underflow, &div_by_zero);
        }

        //
        // Comparator operators
        //

        bool operator<(Unum4 const& other) { return (bool)unum4_lt(val, other.val); }
        bool operator<=(Unum4 const& other) { return (bool)unum4_le(val, other.val); }
        bool operator>(Unum4 const& other) { return (bool)unum4_gt(val, other.val); }
        bool operator>=(Unum4 const& other) { return (bool)unum4_ge(val, other.val); }
        bool operator==(Unum4 const& other) { return (bool)unum4_eq(val, other.val); }
        bool operator!=(Unum4 const& other) { return (bool)!unum4_eq(val, other.val); }

        //
        // Input/Output
        //

        friend std::ostream &operator<<(std::ostream &output, const Unum4 &other ) {
            output << unum42double(other.val);
         return output;
        }

        friend std::istream &operator>>(std::istream  &input, Unum4 &other ) {
            double value;
            input >> value;
            other.val = double2unum4(value, &failed);
            return input;
        }
};

void conversion_tests(){
    std::cout << std::endl << "Conversion tests" << std::endl << std::endl;

    Unum4 u = 37.48;
    std::cout << "u = " << u << std::endl;

    std::cout << "(double)u = " << (double)u << std::endl;
    std::cout << "(float)u = " << (float)u << std::endl;
    std::cout << "(int)u = " << (int)u << std::endl;
    std::cout << std::endl;

    double d = 54.35;
    std::cout << "d = " << d << std::endl;
    std::cout << "(Unum4)d = " << (Unum4)d << std::endl;
    std::cout << std::endl;

    float f = 78.34;
    std::cout << "f = " << f << std::endl;
    std::cout << "(Unum4)f = " << (Unum4)f << std::endl;
    std::cout << std::endl;

    int i = 400;
    std::cout << "i = " << i << std::endl;
    std::cout << "(Unum4)i = " << (Unum4)i << std::endl;
    std::cout << std::endl;

    return;
}

void arithmetic_tests() {
    std::cout << std::endl << "Arithmetic tests" << std::endl << std::endl;

    Unum4 a = 1.0;
    Unum4 b = 2.0F;
    Unum4 c;
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << std::endl;

    std::cout << "c++ = " << c++ << std::endl;
    std::cout << "++c = " << ++c << std::endl;
    c = a + b;
    std::cout << "c = a + b = " << c << std::endl;
    c = b + 4.0;
    std::cout << "c = b + 4.0 = " << c << std::endl;
    c = b + 3.0F;
    std::cout << "c = b + 3.0F = " << c << std::endl;
    c = b + 7;
    std::cout << "c = b + 7 = " << c << std::endl;
    c += b;
    std::cout << "c += b = " << c << std::endl;
    std::cout << std::endl;

    std::cout << "c-- = " << c-- << std::endl;
    std::cout << "--c = " << --c << std::endl;
    c = a - b;
    std::cout << "-b = " << -b << std::endl;
    std::cout << "c = a - b = " << c << std::endl;
    c = b - 4.0;
    std::cout << "c = b - 4.0 = " << c << std::endl;
    c = b - 3.0F;
    std::cout << "c = b - 3.0F = " << c << std::endl;
    c = b - 7;
    std::cout << "c = b - 7 = " << c << std::endl;
    c -= b;
    std::cout << "c -= b = " << c << std::endl;
    std::cout << std::endl;

    c = a * b;
    std::cout << "c = a * b = " << c << std::endl;
    c = b * 4.0;
    std::cout << "c = b * 4.0 = " << c << std::endl;
    c = b * 3.0F;
    std::cout << "c = b * 3.0F = " << c << std::endl;
    c = b * 7;
    std::cout << "c = b * 7 = " << c << std::endl;
    c *= b;
    std::cout << "c *= b = " << c << std::endl;
    std::cout << std::endl;

    c = a / b;
    std::cout << "c = a / b = " << c << std::endl;
    c = b / 4.0;
    std::cout << "c = b / 4.0 = " << c << std::endl;
    c = b / 3.0F;
    std::cout << "c = b / 3.0F = " << c << std::endl;
    c = b / 7;
    std::cout << "c = b / 7 = " << c << std::endl;
    c /= b;
    std::cout << "c /= b = " << c << std::endl;
    std::cout << std::endl;

    return;
}

void comparator_test(Unum4 a, Unum4 b) {

    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << std::endl;

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
    std::cout << std::endl;

    return;
}

void comparator_tests(){
    std::cout << std::endl << "Comparator tests" << std::endl << std::endl;

    Unum4 a = 1;
    Unum4 b = 2;
    comparator_test(a, b);

    a = 2;
    b = 1;
    comparator_test(a, b);

    a = 2;
    b = 2;
    comparator_test(a, b);

    return;
}

int main(int argc, char **argv) {

    conversion_tests();

    arithmetic_tests();

    comparator_tests();

    return 0;
}
