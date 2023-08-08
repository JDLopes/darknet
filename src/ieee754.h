#ifndef IEEE754_H
#define IEEE754_H

#define float2type(a) (a)
#define type2float(a) (a)

#define ZERO 0.0
#define ONE  1.0
#define TWO  2.0

#define add(a, b) ({(a) + (b);})
#define sub(a, b) ({(a) - (b);})
#define mul(a, b) ({(a) * (b);})
#define div(a, b) ({(a) / (b);})

#define neg(a) -(a)

#define lt(a, b) (((a) < (b))? 1: 0)
#define le(a, b) (((a) <= (b))? 1: 0)
#define gt(a, b) (((a) > (b))? 1: 0)
#define ge(a, b) (((a) >= (b))? 1: 0)
#define eq(a, b) (((a) == (b))? 1: 0)

#endif
