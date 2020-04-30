#include<math.h>
#include<omp.h>
#include"init.h"

void init(const unsigned int Np, const unsigned int N, const double h, 
          double* xp, double* yp, double* zp, double* fl, double* gl,
          double* hl, double* Fe, double* Ge, double* He,
          int* firstn, int* nextn, unsigned int* number)
{
  // initialize
  #pragma omp parallel
  {
    #pragma omp for 
    for (unsigned int i = 0; i < Np; ++i) nextn[i] = -1; 
    #pragma omp for 
    for (unsigned int i = 0; i < N * N; ++i) { firstn[i] = -1; number[i] = 0; }
    #pragma omp for
    for (unsigned int i = 0; i < N * N * N; ++i) { Fe[i] = Ge[i] = He[i] = 0; }
  }
  
  // populate particle positions and fill firstn and nextn (no need to parallelize)
  int ii,jj,ind,indn;
  for (unsigned int i = 0; i < Np; ++i) 
  {
    // particles uniformly distributed in [2h,h(N-1)-2h]^3 drand48() * h * (N-1);//3
    xp[i] = 3 * h + (h * (N-1) - 6 * h) * drand48(); 
    yp[i] = 3 * h + (h * (N-1) - 6 * h) * drand48(); 
    zp[i] = 3 * h + (h * (N-1) - 6 * h) * drand48(); 
    // forces uniformly distributed in [-1,1] 
    fl[i] = 2 * drand48() - 1; gl[i] = 2 * drand48() - 1; hl[i] = 2 * drand48() - 1;
    ii = (int) xp[i] / h; jj = (int) yp[i] / h; ind = jj + ii * N;
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
