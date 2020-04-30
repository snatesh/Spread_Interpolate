#include<math.h>
#include<omp.h>
#include"spread_interp.h"

#ifdef __SSE2__
  #define MEM_ALIGN 16
#elif __AVX__
  #define MEM_ALIGN 32
#elif defined(__MIC__)
  #define MEM_ALIGN 64
#endif

void spread_interp(double* xp, double* fl, double* Fe, int* firstn, 
                   int* nextn, unsigned int* number, const unsigned short w, 
                   const double h, const unsigned short N, const bool mode)
{
  const double weight = (mode ? 1 : h * h * h);
  const unsigned short w2 = w * w; const unsigned int N2 = N * N; 
  // loop over w^2 groups of columns
  for (unsigned int izero = 0; izero < w; ++izero)
  {
    for (unsigned int jzero = 0; jzero < w; ++jzero)
    {
      // parallelize over the N^2/w^2 columns in a group
      #pragma omp parallel for
      for (unsigned int ijcount = 0; ijcount < N2/w2; ++ijcount)
      {
        // column indices
        unsigned int jj = jzero + w * (ijcount / (N / w));
        unsigned int ii = izero + w * (ijcount % (N / w));
        
        // find first particle in column(ii,jj) and continue if it's there
        int l = firstn[jj + ii * N];
        if (l >= 0 )
        {
          // global indices of w x w x N subarray influenced column(i,j)
          alignas(MEM_ALIGN) unsigned int indc3D[w2 * N];
          for (int i = 0; i < w; ++i) 
          {
            int i3D = ii + i - w/2 + 1;
            for (int j = 0; j < w; ++j)
            {
              int j3D = jj + j - w/2 + 1;
              for (int k3D = 0; k3D < N; ++k3D)
              {
                indc3D[at(i,j,k3D,w,w)] = pbat(i3D,j3D,k3D,N,N);
              }
            }
          }
          
          // gather eulerian foces for one column into contig mem Fec
          alignas(MEM_ALIGN) double Fec[w2 * N * 3];  
          gather(w2 * N, Fec, Fe, indc3D);
          
          // number of pts in this column, particle indices
          unsigned int npts = number[jj + ii * N], indx[npts];
          for (unsigned int ipt = 0; ipt < npts; ++ipt) {indx[ipt] = l; l = nextn[l];}
          
          // gather lagrangian pts and forces into xpc and flc
          alignas(MEM_ALIGN) double xpc[npts * 3]; 
          alignas(MEM_ALIGN) double flc[npts * 3]; 
          gather(npts, xpc, xp, indx);
          gather(npts, flc, fl, indx);
          //for (unsigned int i = 0; i < npts; ++i) std::cout << flc[i] << std::endl;
          // get the kernel w x w x w kernel weights for each particle in col 
          alignas(MEM_ALIGN) double delta[w2 * w * npts];
          for (int j = 0; j < w; ++j)
          {
            // unwrapped y coordinates
            double fj = ((double) jj + j - w/2 + 1) * h;
            for (int i = 0; i < w; ++i)
            {
              // unwrapped x coordinates
              double fi = ((double) ii + i - w/2 + 1) * h;
              for (int k = 0; k < w; ++k)
              {
                unsigned int m = at(i,j,k,w,w);
                #pragma omp simd // vectorization over particles in col
                for (unsigned int ipt = 0; ipt < npts; ++ipt) 
                {
                  // unwrapped z coordinates
                  double fk = (double) (((int)xpc[2 + 3 * ipt] / h) + k - w/2 + 1) * h;
                  // kernel weights 
                  delta[ipt + m * npts] = deltaf(xpc[3 * ipt] - fi, \
                                                 xpc[1 + 3 * ipt] - fj, \
                                                 xpc[2 + 3 * ipt] - fk) * weight;  

                }
              }
            }
          }
  
          if (mode)
          {         
            // update Eulerian density
            int offset1 = w2 * N;
            for (unsigned int ipt = 0; ipt < npts; ++ipt)
            {
              int m0 = ((int) xpc[2 + 3 * ipt] / h - w/2 + 1);
              int offset0 = w2 * m0;
              #pragma omp simd // vectorize over eulerian pts
              for (int i = 0; i < w2 * w; ++i)
              {
                int k = i / w2 + m0, ic = i + offset0;
                if (k < 0) ic = i + offset0 + offset1;
                if (k >= N) ic = i + offset0 - offset1;
                Fec[3 * ic] += delta[ipt + i * npts] * flc[3 * ipt]; 
                Fec[1 + 3 * ic] += delta[ipt + i * npts] * flc[1 + 3 * ipt]; 
                Fec[2 + 3 * ic] += delta[ipt + i * npts] * flc[2 + 3 * ipt]; 
              }
            }
            // scatter back to global eulerian grid
            scatter(w2 * N, Fec, Fe, indc3D);
          } 
          else
          {
            // interpolate lagrangian density
            int offset1 = w2 * N;
            for (int i = 0; i < w2 * w; ++i)
            {
              #pragma omp simd // vectorize over lagrangian pts 
              for (unsigned int ipt = 0; ipt < npts; ++ipt)
              {
                int m0 = ((int) xpc[2 + 3 * ipt] / h - w/2 + 1);
                int offset0 = w2 * m0;
                int k = i / w2 + m0, ic = i + offset0;
                if (k < 0) ic = i + offset0 + offset1;
                if (k >= N) ic = i + offset0 - offset1;
                flc[3 * ipt] += Fec[3 * ic] * delta[ipt + i * npts]; 
                flc[1 + 3 * ipt] += Fec[1 + 3 * ic] * delta[ipt + i * npts]; 
                flc[2 + 3 * ipt] += Fec[2 + 3 * ic] * delta[ipt + i * npts]; 
              }
            }   
            // scatter back to global lagrangian grid
            scatter(npts, flc, fl, indx);
          }
        } 
      // go to next column in group
      } 
    // go to next group
    }
  }
}    

// Normalized ES kernel for w = 6, beta/w = 1.7305, h =1
#pragma omp declare simd
inline double const deltaf(const double x, const double y, const double z)
{
  return exp(7.9602 * (sqrt(1 - x * x / 9) - 1)) * \
         exp(7.9602 * (sqrt(1 - y * y / 9) - 1)) * \
         exp(7.9602 * (sqrt(1 - z * z / 9) - 1)) / 16.274876371520904;
}

// flattened index into 3D array
#pragma omp declare simd
inline unsigned int const at(unsigned int i, unsigned int j,unsigned int k,\
                             const unsigned int Nx, const unsigned int Ny)
{
  return i + Nx * (j + Ny * k);
}

// flatted index into 3D array, corrected for periodic boundaries
#pragma omp declare simd
inline unsigned int const pbat(int i, int j, int k, const int Nx, const int Ny)
{
  if (i < 0) i = Nx + i; else if (i >= Nx) i = i - Nx; 
  if (j < 0) j = Ny + j; else if (j >= Ny) j = j - Ny; 
  return i + Nx * (j + Ny * k);
}

// flatted index into 3D array, corrected for periodic boundaries
#pragma omp declare simd
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

// scatter data from trg at inds into src
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

