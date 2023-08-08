#ifndef UNUM4_H
#define UNUM4_H

#include <stdint.h>
#include "unum4_hw.h"

union {
    float f;
    unum4 u;
  } OP1;

union {
    float f;
    unum4 u;
  } OP2;

union {
    float f;
    unum4 u;
  } RES;

uint8_t FAILED, OVERFLOW, UNDERFLOW, DIV_BY_ZERO;

#define float2type(a) ({\
      RES.u = float2unum4(a, &FAILED);\
      RES.f;\
    })

#define type2float(a) ({\
      OP1.f = a;\
      RES.u = unum42float(OP1.u);\
      RES.f;\
    })

#define ZERO float2type(0.0)
#define ONE  float2type(1.0)
#define TWO  float2type(2.0)

#define add(a, b) ({\
      OP1.f = a;\
      OP2.f = b;\
      RES.u = unum4_add(OP1.u, OP2.u, &OVERFLOW);\
      RES.f;\
    })

#define sub(a, b) ({\
      OP1.f = a;\
      OP2.f = b;\
      RES.u = unum4_sub(OP1.u, OP2.u, &OVERFLOW);\
      RES.f;\
    })

#define mul(a, b) ({\
      OP1.f = a;\
      OP2.f = b;\
      RES.u = unum4_mul(OP1.u, OP2.u, &OVERFLOW, &UNDERFLOW);\
      RES.f;\
    })

#define div(a, b) ({\
      OP1.f = a;\
      OP2.f = b;\
      RES.u = unum4_div(OP1.u, OP2.u, &OVERFLOW, &UNDERFLOW, &DIV_BY_ZERO);\
      RES.f;\
    })

#define neg(a) sub(ZERO, a)

#define lt(a, b) unum4_lt(a, b)
#define le(a, b) unum4_le(a, b)
#define gt(a, b) unum4_gt(a, b)
#define ge(a, b) unum4_ge(a, b)
#define eq(a, b) unum4_eq(a, b)

#endif
