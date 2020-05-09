#include<chebyshev.h>
#include<math.h>
#include<omp.h>

#ifndef MEM_ALIGN
  #define MEM_ALIGN 16
#endif

/* Clenshaw-curtis nodes cpts and weights cwts,
 * i.e. the Chebyshev points of the second kind,
 * and associated weights. These are such that
 * f(cpts) \dot cwts = int_a^b f(x) dx (they are rescaled to [a,b])
 * The implementation follows that given in ATAP by Trefethen */

void clencurt(double* cpts, double* cwts, const double a, const double b, 
              const unsigned int Np1)
{
  const unsigned int N = Np1 - 1;
  const double H = (b - a) / 2, bpad2 = (b + a) / 2;
  alignas(MEM_ALIGN) double v[N-1];
  const double w1 = (N % 2 ? H / (N * N) : H / (N * N - 1));
  const unsigned int end = (N % 2 ? (N - 1) / 2 + 1 : N / 2);
  #pragma omp simd aligned(cpts: MEM_ALIGN)
  for (unsigned int i = 0; i < Np1; ++i) {cpts[i] = H * cos(M_PI * i / N) + bpad2;}
  #pragma omp simd aligned(v: MEM_ALIGN)
  for (unsigned int i = 0; i < N-1; ++i) {v[i] = 1;}
  
  // now do the weights
  cwts[0] = w1; cwts[N] = w1;
  for (unsigned int k = 1; k < end; ++k)
  {
    #pragma omp simd aligned(v: MEM_ALIGN)
    for (unsigned int ii = 1; ii < N; ++ii)
    {
      v[ii-1] -= 2 * cos(2 * k * M_PI * ii / N) / (4 * k * k - 1);
    }
  }
  
  if (!(N % 2))
  {
    #pragma omp simd aligned(v: MEM_ALIGN)
    for (unsigned int ii = 1; ii < N; ++ii)
    {
      v[ii-1] -= cos(N * M_PI * ii / N) / (N * N - 1);
    }
  }  
  
  #pragma omp simd aligned(cwts: MEM_ALIGN)
  for (unsigned int ii = 1; ii < N; ++ii){cwts[ii] = 2 * H * v[ii-1] / N;}  
}
