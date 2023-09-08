#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "cuda.h"
#include "math.h"

ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
Unum4 activate(Unum4 x, ACTIVATION a);
Unum4 gradient(Unum4 x, ACTIVATION a);
void gradient_array(const Unum4 *x, const int n, const ACTIVATION a, Unum4 *delta);
void activate_array(Unum4 *x, const int n, const ACTIVATION a);
#ifdef GPU
void activate_array_gpu(Unum4 *x, int n, ACTIVATION a);
void gradient_array_gpu(Unum4 *x, int n, ACTIVATION a, Unum4 *delta);
#endif

static inline Unum4 stair_activate(Unum4 x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline Unum4 hardtan_activate(Unum4 x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline Unum4 linear_activate(Unum4 x){return x;}
static inline Unum4 logistic_activate(Unum4 x){return 1./(1. + exp(-x));}
static inline Unum4 loggy_activate(Unum4 x){return 2./(1. + exp(-x)) - 1;}
static inline Unum4 relu_activate(Unum4 x){return x*(x>0);}
static inline Unum4 elu_activate(Unum4 x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline Unum4 selu_activate(Unum4 x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);}
static inline Unum4 relie_activate(Unum4 x){return (x>0) ? x : .01*x;}
static inline Unum4 ramp_activate(Unum4 x){return x*(x>0)+.1*x;}
static inline Unum4 leaky_activate(Unum4 x){return (x>0) ? x : .1*x;}
static inline Unum4 tanh_activate(Unum4 x){return (exp(2*x)-1)/(exp(2*x)+1);}
static inline Unum4 plse_activate(Unum4 x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline Unum4 lhtan_activate(Unum4 x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
static inline Unum4 lhtan_gradient(Unum4 x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline Unum4 hardtan_gradient(Unum4 x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline Unum4 linear_gradient(Unum4 x){return 1;}
static inline Unum4 logistic_gradient(Unum4 x){return (1-x)*x;}
static inline Unum4 loggy_gradient(Unum4 x)
{
    Unum4 y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline Unum4 stair_gradient(Unum4 x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static inline Unum4 relu_gradient(Unum4 x){return (x>0);}
static inline Unum4 elu_gradient(Unum4 x){return (x >= 0) + (x < 0)*(x + 1);}
static inline Unum4 selu_gradient(Unum4 x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
static inline Unum4 relie_gradient(Unum4 x){return (x>0) ? 1 : .01;}
static inline Unum4 ramp_gradient(Unum4 x){return (x>0)+.1;}
static inline Unum4 leaky_gradient(Unum4 x){return (x>0) ? 1 : .1;}
static inline Unum4 tanh_gradient(Unum4 x){return 1-x*x;}
static inline Unum4 plse_gradient(Unum4 x){return (x < 0 || x > 1) ? .01 : .125;}

#endif

