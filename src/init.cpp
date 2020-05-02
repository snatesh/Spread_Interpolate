#include<math.h>
#include<omp.h>
#include"init.h"

// no pbc
void init(const unsigned int Np, const unsigned int N, const double h, double* xp, double* fl, double* Fe, 
          int* firstn, int* nextn, unsigned int* number)
{
  // initialize
  #pragma omp parallel
  {
    #pragma omp for 
    for (unsigned int i = 0; i < Np; ++i) nextn[i] = -1; 
    #pragma omp for 
    for (unsigned int i = 0; i < N * N; ++i) { firstn[i] = -1; number[i] = 0;}
    #pragma omp for
    for (unsigned int i = 0; i < N * N * N * 3; ++i) Fe[i] = 0;
  }
  
  // populate particle positions and fill firstn and nextn (no need to parallelize)
  int ii,jj,ind,indn;
  for (unsigned int i = 0; i < Np; ++i) 
  {

    for (unsigned int j = 0; j < 3; ++j)
    {//drand48() * h * (N-1); //
      // particles uniformly distributed in [2h,h(N-1)-2h]^3 drand48() * h * (N-1);//3
      xp[j + 3 * i] = 3 * h + (h * (N-1) - 6 * h) * drand48(); 
      // forces uniformly distributed in [-1,1] 
      fl[j + 3 * i] = 2 * drand48() - 1;
    }
    ii = (int) xp[3 * i] / h; jj = (int) xp[1 + 3 * i] / h; ind = jj + ii * N;
    if (firstn[ind] < 0) { firstn[ind] = i;}
    else
    {
      indn = firstn[ind];
      while (nextn[indn] >= 0)
      {
        indn = nextn[indn];
      }
      nextn[indn] = i;
    }
    number[ind] += 1;
  }
}

// with pbc N = Nwrap + w
void init(const unsigned int Np, const unsigned int N, const unsigned short w, 
          const double h, double* xp, double* fl, double* Fe, 
          int* firstn, int* nextn, unsigned int* number)
{
  const unsigned int Nwrap = N - w;
  // initialize
  #pragma omp parallel
  {
    #pragma omp for 
    for (unsigned int i = 0; i < Np; ++i) nextn[i] = -1; 
    #pragma omp for 
    for (unsigned int i = 0; i < N * N; ++i) { firstn[i] = -1; number[i] = 0;}
    #pragma omp for
    for (unsigned int i = 0; i < N * N * N * 3; ++i) Fe[i] = 0;
  }
  
  // populate particle positions and fill firstn and nextn (no need to parallelize)
  int ii,jj,ind,indn;
  for (unsigned int i = 0; i < Np; ++i) 
  {

    for (unsigned int j = 0; j < 3; ++j)
    {
      // particles uniformly distributed in [0,h(N-1)]^3
      // note: unwrapped grid is [-wh/2,h(N-1)+wh/2]^3 
      xp[j + 3 * i] = drand48() * h * (Nwrap-1); 
      // forces uniformly distributed in [-1,1] 
      fl[j + 3 * i] = 10;//2 * drand48() - 1;
    }
    // when we use column indices for coordinate calculation, must subtract w/2
    ii = (int) xp[3 * i] / h + w / 2; jj = (int) xp[1 + 3 * i] / h + w / 2; ind = jj + ii * N;
    if (firstn[ind] < 0) { firstn[ind] = i;}
    else
    {
      indn = firstn[ind];
      while (nextn[indn] >= 0)
      {
        indn = nextn[indn];
      }
      nextn[indn] = i;
    }
    number[ind] += 1;
  }
}

// lagrangian and eulerian forces, positions read from file
void init(const unsigned int Np, const unsigned int N, const double h, double* xp,
          int* firstn, int* nextn, unsigned int* number)
{
  // initialize
  #pragma omp parallel
  {
    #pragma omp for 
    for (unsigned int i = 0; i < Np; ++i) nextn[i] = -1; 
    #pragma omp for 
    for (unsigned int i = 0; i < N * N; ++i) { firstn[i] = -1; number[i] = 0;}
  }
  
  // populate particle positions and fill firstn and nextn (no need to parallelize)
  int ii,jj,ind,indn;
  for (unsigned int i = 0; i < Np; ++i) 
  {
    ii = (int) xp[3 * i] / h; jj = (int) xp[1 + 3 * i] / h; ind = jj + ii * N;
    if (firstn[ind] < 0) { firstn[ind] = i;}
    else
    {
      indn = firstn[ind];
      while (nextn[indn] >= 0)
      {
        indn = nextn[indn];
      }
      nextn[indn] = i;
    }
    number[ind] += 1;
  }
}
