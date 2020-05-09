#ifndef CHEBYSHEV_H
#define CHEBYSHEV_H

/* Clenshaw-curtis nodes cpts and weights cwts,
 * i.e. the Chebyshev points of the second kind,
 * and associated weights. These are such that
 * f(cpts) \dot cwts = int_a^b f(x) dx.
 * The implementation follows that given in ATAP by Trefethen */

void clencurt(double* cpts, double* cwts, const double a, const double b, 
              const unsigned int Np1);



#endif
