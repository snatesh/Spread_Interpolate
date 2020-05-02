#ifndef INIT_H
#define INIT_H

// no pbc
void init(const unsigned int Np, const unsigned int N, const double h, double* xp, double* fl, double* Fe, 
          int* firstn, int* nextn, unsigned int* number);

// read from fl and Fe from file
void init(const unsigned int Np, const unsigned int N, const double h, double* xp,
          int* firstn, int* nextn, unsigned int* number);
// with pbc
void init(const unsigned int Np, const unsigned int N, const unsigned short w, 
          const double h, double* xp, double* fl, double* Fe, 
          int* firstn, int* nextn, unsigned int* number);
// positions read from file with pbc
void init(const unsigned int Np, const unsigned int N, const unsigned short w, 
          const double h, double* xp, int* firstn, int* nextn, unsigned int* number);
#endif
