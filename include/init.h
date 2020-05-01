#ifndef INIT_H
#define INIT_H

void init(const unsigned int Np, const unsigned int N, const double h, double* xp, double* fl, double* Fe, 
          int* firstn, int* nextn, unsigned int* number);

void init(const unsigned int Np, const unsigned int N, const double h, double* xp,
          int* firstn, int* nextn, unsigned int* number);
#endif
