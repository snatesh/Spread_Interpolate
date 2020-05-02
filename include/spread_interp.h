#ifndef _SPREAD_INTERP_H
#define _SPREAD_INTERP_H

#include<math.h>
#ifdef DEBUG
#include<iostream>
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

// get delta function weights for particles in a given column
void get_delta_col(double* delta, const double* xpc, const double fi, 
                   const double fj, const double weight, const int m, 
                   const int k, const int npts, const int w, const int h);
// spread forces on particles in a given column to local eulerian grid
void spread_col(double* Fec, const double* delta, const double* flc, 
                const int i0, const int ipt, const int npts, const int w3);
// interpolate data around particles in a column to the particles
void interp_col(const double* Fec, const double* delta, double* flc, 
                const int i0, const int ipt, const int npts, const int w3);

// implements copy opertion to enforce periodicity of eulerian data before interpolation
void copy_pbc(double* Fe, const double* Fe_wrap, const unsigned short w, const unsigned int N);
// implements fold operation to de-ghostify spread data, i.e. enable periodic spread
void fold_pbc(double* Fe, double* Fe_wrap, const unsigned short w, const unsigned int N);

#endif
