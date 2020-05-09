#ifndef _SPREAD_INTERP_H
#define _SPREAD_INTERP_H

#include<math.h>
#ifdef DEBUG
#include<iostream>
#endif

#if defined(__MIC__)
  #define MEM_ALIGN 64
#elif __AVX__
  #define MEM_ALIGN 32
#elif __SSE2__
  #define MEM_ALIGN 16
#endif

// Normalized ES kernel for w = 6, beta/w = 1.7305, h =1
#pragma omp declare simd
inline double const deltaf(const double x[3])
{
  return exp(7.9602 * (sqrt(1 - x[0] * x[0] / 9) - 1)) * \
         exp(7.9602 * (sqrt(1 - x[1] * x[1] / 9) - 1)) * \
         exp(7.9602 * (sqrt(1 - x[2] * x[2] / 9) - 1)) / 16.274876371520904;
}

#pragma omp declare simd
inline double const deltaf(const double x, const double y, const double z)
{
  return exp(7.9602 * (sqrt(1 - x * x / 9) - 1)) * \
         exp(7.9602 * (sqrt(1 - y * y / 9) - 1)) * \
         exp(7.9602 * (sqrt(1 - z * z / 9) - 1)) / 16.274876371520904;
}

// flattened index into 3D array
inline unsigned int const at(unsigned int i, unsigned int j,unsigned int k,\
                             const unsigned int Nx, const unsigned int Ny)
{
  return i + Nx * (j + Ny * k);
}

// flatted index into 3D array, corrected for periodic boundaries
inline unsigned int const pbat(int i, int j, int k, const int Nx, const int Ny)
{
  if (i < 0) i = Nx + i; else if (i >= Nx) i = i - Nx; 
  if (j < 0) j = Ny + j; else if (j >= Ny) j = j - Ny; 
  return i + Nx * (j + Ny * k);
}

// flatted index into 3D array, corrected for periodic boundaries
inline unsigned int const pbat(int i, int j, int k, const int Nx, const int Ny, const int Nz)
{
  if (i < 0) i = Nx + i; else if (i >= Nx) i = i - Nx; 
  if (j < 0) j = Ny + j; else if (j >= Ny) j = j - Ny; 
  if (k < 0) k = Nz + k; else if (k >= Nz) k = k - Nz;
  return i + Nx * (j + Ny * k);
}


// gather data from src at inds into trg
inline void gather(unsigned int N, double* trg, double const* src, const unsigned int* inds)
{
  #pragma omp simd
  for (unsigned int i = 0; i < N; ++i) 
  {
    trg[3 * i] = src[3 * inds[i]];
    trg[1 + 3 * i] = src[1 + 3 * inds[i]];
    trg[2 + 3 * i] = src[2 + 3 * inds[i]];
  }
}


// scatter data from trg into src at inds
inline void scatter(unsigned int N, double const* trg, double* src, const unsigned int* inds)
{
  #pragma omp simd
  for (unsigned int i = 0; i < N; ++i) 
  {
    src[3 * inds[i]] = trg[3 * i];
    src[1 + 3 * inds[i]] = trg[1 + 3 * i];
    src[2 + 3 * inds[i]] = trg[2 + 3 * i];
  }
}

inline void gather(unsigned int N, double* trgx, double* trgy, double* trgz, 
                   double const* src, const unsigned int* inds)
{
  #pragma omp simd
  for (unsigned int i = 0; i < N; ++i) 
  {
    trgx[i] = src[3 * inds[i]];
    trgy[i] = src[1 + 3 * inds[i]];
    trgz[i] = src[2 + 3 * inds[i]];
  }
}

// scatter data from trg into src at inds
inline void scatter(unsigned int N, double const* trgx, double const* trgy, 
                    double const* trgz, double* src, const unsigned int* inds)
{
  #pragma omp simd
  for (unsigned int i = 0; i < N; ++i) 
  {
    src[3 * inds[i]] = trgx[i];
    src[1 + 3 * inds[i]] = trgy[i];
    src[2 + 3 * inds[i]] = trgz[i];
  }
}

// spreading (mode=true) and interpolation (mode=false) with pbc
// calculations done on the fly (bad for vectorization)
void spread_interp(double* xp, double* fl, double* Fe, int* firstn, 
                   int* nextn, unsigned int* number, const unsigned short w, 
                   const double h, const unsigned short N, const bool mode);

// spreading (mode=true) and interpolation (mode=false) 
// with pbc corrections done separately
void spread_interp_pbc(double* xp, double* fl, double* Fe, double* Fe_wrap, int* firstn, 
                   int* nextn, unsigned int* number, const unsigned short w, 
                   const double h, const unsigned short N, const bool mode);
#ifndef SEPXYZ
// get delta function weights for particles in a given column
void delta_col(double* delta, const double* xpc, const double fi, 
                   const double fj, const double weight, const int m, 
                   const int k, const int npts, const int w, const int h);
// spread forces on particles in a given column to local eulerian grid
void spread_col(double* Fec, const double* delta, const double* flc, 
                const int i0, const int ipt, const int npts, const int w3);
// interpolate data around particles in a column to the particles
void interp_col(const double* Fec, const double* delta, double* flc, 
                const int i0, const int ipt, const int npts, const int w3);
#else
// get delta function weights for particles in a given column
inline void delta_col(double* delta, const double* xpc, const double* ypc,
                   const double* zpc, const double fi, const double fj, 
                   const double weight, const int m, const int k, const int npts, 
                   const int w, const int w3, const int h)
{
  #pragma omp simd aligned(delta,xpc,ypc,zpc: MEM_ALIGN) 
  for (unsigned int ipt = 0; ipt < npts; ++ipt) 
  {
   // unwrapped z coordinates
    double fk = (double) (((int)zpc[ipt] / h) + k - w/2 + 1) * h;
    alignas(MEM_ALIGN) double x[3]; 
    x[0] = xpc[ipt] - fi;
    x[1] = ypc[ipt] - fj;
    x[2] = zpc[ipt] - fk;
    // kernel weights 
    delta[m + ipt * w3] = deltaf(x) * weight;  
  }
}

inline void delta_col1(double* x, const int w3)
{
  #pragma omp simd aligned(x: MEM_ALIGN)
  for (unsigned int i = 0; i < w3; ++i)
  {
    x[i] = 1 - x[i] * x[i] / 9;
  }
}

inline void delta_col2(double* x, const int w3)
{
  #pragma omp simd aligned(x: MEM_ALIGN)
  for (unsigned int i = 0; i < w3; ++i)
  {
    x[i] = 7.9602 * (sqrt(x[i]) - 1);
  } 
}

inline void delta_col3(double* x, const int w3)
{
  #pragma omp simd aligned(x: MEM_ALIGN)
  for (unsigned int i = 0; i < w3; ++i)
  {
    x[i] = exp(x[i]); 
  } 
}

// get delta function weights for particles in a given column
inline void delta_col(double* delta, const double* x, const double* y,
    const double* z, const double weight, const int w3)
{
  #pragma omp simd aligned(delta,x,y,z: MEM_ALIGN) 
  for (unsigned int i = 0; i < w3; ++i) 
  {
    delta[i] = weight * x[i] * y[i] * z[i] / 16.274876371520904;
  }
}

// spread forces on particles in a given column to local eulerian grid
inline void spread_col(double* Fec, double* Gec, double* Hec, const double* delta, 
                const double flc, const double glc, const double hlc, const int w3)
{
  #pragma omp simd aligned(Fec,Gec,Hec,delta: MEM_ALIGN)
  for (unsigned int i = 0; i < w3; ++i)
  {
    Fec[i] = Fec[i] + delta[i] * flc;
    Gec[i] = Gec[i] + delta[i] * glc;
    Hec[i] = Hec[i] + delta[i] * hlc;
  }
}

// interpolate data around particles in a column to the particles
inline void interp_col(const double* Fec, const double* Gec, const double* Hec,
                const double* delta, double* flc, double* glc, double* hlc,
                const int ipt, const int w3)
{
  double flsum, glsum, hlsum; flsum = glsum = hlsum = 0;
  #pragma omp simd aligned(Fec,Gec,Hec,delta: MEM_ALIGN) reduction(+:flsum,glsum,hlsum) 
  for (unsigned int i = 0; i < w3; ++i)
  {
    flsum += Fec[i] * delta[i]; 
    glsum += Gec[i] * delta[i]; 
    hlsum += Hec[i] * delta[i]; 
  }
  flc[ipt] += flsum; glc[ipt] += glsum; hlc[ipt] += hlsum;
}

#endif

// implements copy opertion to enforce periodicity of eulerian data before interpolation
inline void copy_pbc(double* Fe, const double* Fe_wrap, const unsigned short w, const unsigned int N)
{
  unsigned int lend = w / 2, Nwrap = N - w;
  unsigned int rbeg = N - lend; 
  #pragma omp parallel
  {
    // copy data on wrapped grid to extended periodic grid
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nwrap; ++k)
    {
      for (unsigned int j = 0; j < Nwrap; ++j)
      {
        //#pragma omp simd aligned(Fe, Fe_wrap: MEM_ALIGN)
        for (unsigned int i = 0; i < Nwrap; ++i)
        {
          unsigned int ii = i + lend, jj = j + lend, kk = k + lend;
          Fe[3 * at(ii, jj, kk, N, N)] = Fe_wrap[3 * at(i, j, k, Nwrap, Nwrap)];
          Fe[1 + 3 * at(ii, jj, kk, N, N)] = Fe_wrap[1 + 3 * at(i, j, k, Nwrap, Nwrap)];
          Fe[2 + 3 * at(ii, jj, kk, N, N)] = Fe_wrap[2 + 3 * at(i, j, k, Nwrap, Nwrap)];
        }
      }
    }
  
    // copy eulerian data in y-z plane in periodic region to ghost
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < N; ++k)
    {
      for (unsigned int j = 0; j < N; ++j)
      {
        // first copy left to right
        for (unsigned int i = rbeg; i < N; ++i)
        {
          unsigned int ipb = i - Nwrap;
          Fe[3 * at(i, j, k, N, N)] = Fe[3 * at(ipb, j, k, N, N)]; 
          Fe[1 + 3 * at(i, j, k, N, N)] = Fe[1 + 3 * at(ipb, j, k, N, N)]; 
          Fe[2 + 3 * at(i, j, k, N, N)] = Fe[2 + 3 * at(ipb, j, k, N, N)]; 
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < N; ++k)
    {
      for (unsigned int j = 0; j < N; ++j)
      {
        // now copy right to left
        for (unsigned int i = 0; i < lend; ++i)
        {
          unsigned int ipb = i + Nwrap;
          Fe[3 * at(i, j, k, N, N)] += Fe[3 * at(ipb, j, k, N, N)]; 
          Fe[1 + 3 * at(i, j, k, N, N)] += Fe[1 + 3 * at(ipb, j, k, N, N)]; 
          Fe[2 + 3 * at(i, j, k, N, N)] += Fe[2 + 3 * at(ipb, j, k, N, N)]; 
        }  
      }
    }
    // copy eulerian data in x-z plane in periodic region to ghost
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < N; ++k)
    {
      for (unsigned int j = rbeg; j < N; ++j)
      {
        // first copy bottom to top
        for (unsigned int i = 0; i < N; ++i)
        {
          unsigned int jpb = j - Nwrap;
          Fe[3 * at(i, j, k, N, N)] = Fe[3 * at(i, jpb, k, N, N)]; 
          Fe[1 + 3 * at(i, j, k, N, N)] = Fe[1 + 3 * at(i, jpb, k, N, N)]; 
          Fe[2 + 3 * at(i, j, k, N, N)] = Fe[2 + 3 * at(i, jpb, k, N, N)]; 
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < N; ++k)
    {
      for (unsigned int j = 0; j < lend; ++j)
      {
        // now copy top to bottom
        for (unsigned int i = 0; i < N; ++i)
        {
          unsigned int jpb = j + Nwrap;
          Fe[3 * at(i, j, k, N, N)] += Fe[3 * at(i, jpb, k, N, N)]; 
          Fe[1 + 3 * at(i, j, k, N, N)] += Fe[1 + 3 * at(i, jpb, k, N, N)]; 
          Fe[2 + 3 * at(i, j, k, N, N)] += Fe[2 + 3 * at(i, jpb, k, N, N)]; 
        }  
      }
    }
    // copy eulerian data in x-y plane in periodic region to ghost
    #pragma omp for collapse(3)
    for (unsigned int k = rbeg; k < N; ++k)
    {
      for (unsigned int j = 0; j < N; ++j)
      {
        // first copy down to up
        for (unsigned int i = 0; i < N; ++i)
        {
          unsigned int kpb = k - Nwrap;
          Fe[3 * at(i, j, k, N, N)] = Fe[3 * at(i, j, kpb, N, N)]; 
          Fe[1 + 3 * at(i, j, k, N, N)] = Fe[1 + 3 * at(i, j, kpb, N, N)]; 
          Fe[2 + 3 * at(i, j, k, N, N)] = Fe[2 + 3 * at(i, j, kpb, N, N)]; 
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < lend; ++k)
    {
      for (unsigned int j = 0; j < N; ++j)
      {
        // now copy top to bottom
        for (unsigned int i = 0; i < N; ++i)
        {
          unsigned int kpb = k + Nwrap;
          Fe[3 * at(i, j, k, N, N)] += Fe[3 * at(i, j, kpb, N, N)]; 
          Fe[1 + 3 * at(i, j, k, N, N)] += Fe[1 + 3 * at(i, j, kpb, N, N)]; 
          Fe[2 + 3 * at(i, j, k, N, N)] += Fe[2 + 3 * at(i, j, kpb, N, N)]; 
        }
      }
    }
  }
}

// implements fold operation to de-ghostify spread data, i.e. enable periodic spread
inline void fold_pbc(double* Fe, double* Fe_wrap, const unsigned short w, const unsigned int N)
{
  unsigned int lend = w / 2, Nwrap = N - w;
  unsigned int rbeg = N - lend; 
  #pragma omp parallel
  {
    // fold eulerian data in y-z plane in ghost region to periodic index
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < N; ++k)
    {
      for (unsigned int j = 0; j < N; ++j)
      {
        // first do left
        for (unsigned int i = 0; i < lend; ++i)
        {
          unsigned int ipb = i + Nwrap;
          Fe[3 * at(ipb, j, k, N, N)] += Fe[3 * at(i, j, k, N, N)]; 
          Fe[1 + 3 * at(ipb, j, k, N, N)] += Fe[1 + 3 * at(i, j, k, N, N)]; 
          Fe[2 + 3 * at(ipb, j, k, N, N)] += Fe[2 + 3 * at(i, j, k, N, N)]; 
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < N; ++k)
    {
      for (unsigned int j = 0; j < N; ++j)
      {
        // now do right
        for (unsigned int i = rbeg; i < N; ++i)
        {
          unsigned int ipb = i - Nwrap;
          Fe[3 * at(ipb, j, k, N, N)] += Fe[3 * at(i, j, k, N, N)]; 
          Fe[1 + 3 * at(ipb, j, k, N, N)] += Fe[1 + 3 * at(i, j, k, N, N)]; 
          Fe[2 + 3 * at(ipb, j, k, N, N)] += Fe[2 + 3 * at(i, j, k, N, N)]; 
        }  
      }
    }
    // fold eulerian data in x-z plane in ghost region to periodic index
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < N; ++k)
    {
      // first do bottom
      for (unsigned int j = 0; j < lend; ++j)
      {
        for (unsigned int i = 0; i < N; ++i)
        {
          unsigned int jpb = j + Nwrap;
          Fe[3 * at(i, jpb, k, N, N)] += Fe[3 * at(i, j, k, N, N)]; 
          Fe[1 + 3 * at(i, jpb, k, N, N)] += Fe[1 + 3 * at(i, j, k, N, N)]; 
          Fe[2 + 3 * at(i, jpb, k, N, N)] += Fe[2 + 3 * at(i, j, k, N, N)]; 
        }
      } 
    } 
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < N; ++k)
    {
      // now do top
      for (unsigned int j = rbeg; j < N; ++j)
      {
        for (unsigned int i = 0; i < N; ++i)
        {
          unsigned int jpb = j - Nwrap;
          Fe[3 * at(i, jpb, k, N, N)] += Fe[3 * at(i, j, k, N, N)]; 
          Fe[1 + 3 * at(i, jpb, k, N, N)] += Fe[1 + 3 * at(i, j, k, N, N)]; 
          Fe[2 + 3 * at(i, jpb, k, N, N)] += Fe[2 + 3 * at(i, j, k, N, N)]; 
        }
      }
    }
    // fold eulerian data in x-y plane in ghost region to periodic index
    // first do down
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < lend; ++k)
    {
      for (unsigned int j = 0; j < N; ++j)
      {
        for (unsigned int i = 0; i < N; ++i)
        {
          unsigned int kpb = k + Nwrap;
          Fe[3 * at(i, j, kpb, N, N)] += Fe[3 * at(i, j, k, N, N)]; 
          Fe[1 + 3 * at(i, j, kpb, N, N)] += Fe[1 + 3 * at(i, j, k, N, N)]; 
          Fe[2 + 3 * at(i, j, kpb, N, N)] += Fe[2 + 3 * at(i, j, k, N, N)]; 
        }
      } 
    } 
  
    // now do up
    #pragma omp for collapse(3)
    for (unsigned int k = rbeg; k < N; ++k)
    {
      for (unsigned int j = 0; j < N; ++j)
      {
        for (unsigned int i = 0; i < N; ++i)
        {
          unsigned int kpb = k - Nwrap;
          Fe[3 * at(i, j, kpb, N, N)] += Fe[3 * at(i, j, k, N, N)]; 
          Fe[1 + 3 * at(i, j, kpb, N, N)] += Fe[1 + 3 * at(i, j, k, N, N)]; 
          Fe[2 + 3 * at(i, j, kpb, N, N)] += Fe[2 + 3 * at(i, j, k, N, N)]; 
        }
      }
    }
    // copy data on extended periodic grid to wrapped grid
    #pragma omp for collapse(3)
    for (unsigned int k = lend; k < rbeg; ++k) 
    {
      for (unsigned int j = lend; j < rbeg; ++j)
      {
        for (unsigned int i = lend; i < rbeg; ++i)
        {
          unsigned int ii = i - lend, jj = j - lend, kk = k - lend;
          Fe_wrap[3 * at(ii, jj, kk, Nwrap, Nwrap)] = Fe[3 * at(i, j, k, N, N)];
          Fe_wrap[1 + 3 * at(ii, jj, kk, Nwrap, Nwrap)] = Fe[1 + 3 * at(i, j, k, N, N)];
          Fe_wrap[2 + 3 * at(ii, jj, kk, Nwrap, Nwrap)] = Fe[2 + 3 * at(i, j, k, N, N)];
        }
      }
    }
  }
}

#endif
