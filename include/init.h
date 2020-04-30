#ifndef INIT_H
#define INIT_H

void init(const unsigned int Np, const unsigned int N, const double h, 
          double* xp, double* yp, double* zp, double* fl, double* gl,
          double* hl, double* Fe, double* Ge, double* He,
          int* firstn, int* nextn, unsigned int* number);
#endif
