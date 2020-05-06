#include<omp.h>
#include"spread_interp.h"
#ifdef DEBUG
#include<iostream>
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

void spread_interp_pbc(double* xp, double* fl, double* Fe, double* Fe_wrap, int* firstn, 
                   int* nextn, unsigned int* number, const unsigned short w, 
                   const double h, const unsigned short N, const bool mode)
{
  const double weight = (mode ? 1 : h * h * h);
  const unsigned short w2 = w * w; const unsigned int N2 = N * N; 
  const unsigned short w3 = w2 * w;
  // ensure periodicity of eulerian data for interpolation
  if (!mode) copy_pbc(Fe, Fe_wrap, w, N);

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
          for (int k3D = 0; k3D < N; ++k3D)
          {
            for (int j = 0; j < w; ++j)
            {
              int j3D = jj + j - w/2 + 1;
              for (int i = 0; i < w; ++i) 
              {
                int i3D = ii + i - w/2 + 1;
                  indc3D[at(i,j,k3D,w,w)] = at(i3D,j3D,k3D,N,N);
              }
            }
          }
          // gather eulerian foces for one column into contig mem Fec
          #ifndef SEPXYZ
            alignas(MEM_ALIGN) double Fec[w2 * N * 3];  
            gather(w2 * N, Fec, Fe, indc3D);
          #else
            alignas(MEM_ALIGN) double Fec[w2 * N];  
            alignas(MEM_ALIGN) double Gec[w2 * N];  
            alignas(MEM_ALIGN) double Hec[w2 * N];  
            gather(w2 * N, Fec, Gec, Hec, Fe, indc3D);
          #endif 

          
          // number of pts in this column, particle indices
          unsigned int npts = number[jj + ii * N];
          alignas(MEM_ALIGN) unsigned int indx[npts];
          for (unsigned int ipt = 0; ipt < npts; ++ipt) {indx[ipt] = l; l = nextn[l];}
          
          // gather lagrangian pts and forces into xpc and flc
          #ifndef SEPXYZ 
            alignas(MEM_ALIGN) double xpc[npts * 3]; 
            alignas(MEM_ALIGN) double flc[npts * 3]; 
            gather(npts, xpc, xp, indx);
            gather(npts, flc, fl, indx);
          #else
            alignas(MEM_ALIGN) double xpc[npts]; 
            alignas(MEM_ALIGN) double ypc[npts]; 
            alignas(MEM_ALIGN) double zpc[npts]; 
            alignas(MEM_ALIGN) double flc[npts]; 
            alignas(MEM_ALIGN) double glc[npts]; 
            alignas(MEM_ALIGN) double hlc[npts]; 
            gather(npts, xpc, ypc, zpc, xp, indx);
            gather(npts, flc, glc, hlc, fl, indx);
          #endif
          // get the kernel w x w x w kernel weights for each particle in col 
          alignas(MEM_ALIGN) double delta[w2 * w * npts];
          #ifndef SEPXYZ
          for (int k = 0; k < w; ++k)
          {
            for (int j = 0; j < w; ++j)
            {
              // unwrapped y coordinates (correcting jj with sub w/2)
              double fj = ((double) jj + j - w + 1) * h;
              for (int i = 0; i < w; ++i)
              {
                // unwrapped x coordinates (correcting ii with sub w/2)
                double fi = ((double) ii + i - w + 1) * h;

                unsigned int m = at(i,j,k,w,w);
                delta_col(delta, xpc, fi, fj, weight, m, k, npts, w, h);
              }
            }
          }
          #else

          //for (int k = 0; k < w; ++k)
          //{
          //  for (int j = 0; j < w; ++j)
          //  {
          //    // unwrapped y coordinates (correcting jj with sub w/2)
          //    double fj = ((double) jj + j - w + 1) * h;
          //    for (int i = 0; i < w; ++i)
          //    {
          //      // unwrapped x coordinates (correcting ii with sub w/2)
          //      double fi = ((double) ii + i - w + 1) * h;

          //      unsigned int m = at(i,j,k,w,w);
          //      delta_col(delta, xpc, ypc, zpc, fi, fj, weight, m, k, npts, w, w3, h);  
          //    }
          //  }
          //}
          alignas(MEM_ALIGN) double xd[w3];
          alignas(MEM_ALIGN) double yd[w3];
          alignas(MEM_ALIGN) double zd[w3];
          for (unsigned int ipt = 0; ipt < npts; ++ipt) 
          {
            #pragma omp simd collapse(3)
            for (int k = 0; k < w; ++k)
            {
              for (int j = 0; j < w; ++j)
              {
                for (int i = 0; i < w; ++i)
                {
                  // unwrapped x coordinates (correcting ii with sub w/2)
                  double fi = ((double) ii + i - w + 1) * h;
                  // unwrapped y coordinates (correcting jj with sub w/2)
                  double fj = ((double) jj + j - w + 1) * h;
                  // unwrapped z coordinates
                  double fk = (double) (((int)zpc[ipt] / h) + k - w/2 + 1) * h;
                  unsigned int m = at(i,j,k,w,w);
                  xd[m] = xpc[ipt] - fi;  
                  yd[m] = ypc[ipt] - fj;  
                  zd[m] = zpc[ipt] - fk;  
                }
              }
            }
            delta_col1(xd, w3); delta_col1(yd, w3); delta_col1(zd, w3);
            delta_col2(xd, w3); delta_col2(yd, w3); delta_col2(zd, w3);
            delta_col3(xd, w3); delta_col3(yd, w3); delta_col3(zd, w3);
            delta_col(&delta[ipt * w3], xd, yd, zd, weight, w3);
          }
          #endif
  
          if (mode)
          {         
            // update Eulerian density
            #ifndef SEPXYZ
            for (unsigned int ipt = 0; ipt < npts; ++ipt)
            {
                int i0 = w2 * ((int) xpc[2 + 3 * ipt] / h + 1);
                spread_col(Fec, delta, flc, i0, ipt, npts, w3);

            }
            // scatter back to global eulerian grid
            scatter(w2 * N, Fec, Fe, indc3D);
            #else
            for (unsigned int ipt = 0; ipt < npts; ++ipt)
            {
                int i0 = w2 * ((int) zpc[ipt] / h + 1);
                spread_col(&Fec[i0], &Gec[i0], &Hec[i0], &delta[ipt * w3],
                           flc[ipt], glc[ipt], hlc[ipt], w3);
            }
            // scatter back to global eulerian grid
            scatter(w2 * N, Fec, Gec, Hec, Fe, indc3D);
            #endif
          } 
          else
          {
            for (unsigned int ipt = 0; ipt < npts; ++ipt)
            {
              #ifndef SEPXYZ
              int i0 = w2 * ((int) xpc[2 + 3 * ipt] / h + 1);
              interp_col(Fec, delta, flc, i0, ipt, npts, w3);
              #else
              int i0 = w2 * ((int) zpc[ipt] / h + 1);
              interp_col(&Fec[i0], &Gec[i0], &Hec[i0], &delta[ipt * w3], 
                         flc, glc, hlc, ipt, w3);
              #endif
            }
            // scatter back to global lagrangian grid
            #ifndef SEPXYZ
              scatter(npts, flc, fl, indx);
            #else
              scatter(npts, flc, glc, hlc, fl, indx);
            #endif
          }
        }
      // go to next column in group
      }
    // go to next group
    }
  }
  // fold periodic spread data from ghost region into interior
  if (mode) fold_pbc(Fe, Fe_wrap, w, N);
}

#ifdef SEPXYZ

#else
void delta_col(double* delta, const double* xpc, const double fi, 
                   const double fj, const double weight, const int m, 
                   const int k, const int npts, const int w, const int h)
{
  #pragma omp simd aligned(delta,xpc: MEM_ALIGN) // vectorization over particles in col
  for (unsigned int ipt = 0; ipt < npts; ++ipt) 
  {
   // unwrapped z coordinates
    double fk = (double) (((int)xpc[2 + 3 * ipt] / h) + k - w/2 + 1) * h;
    alignas(MEM_ALIGN) double x[3]; 
    x[0] = xpc[3 * ipt] - fi;
    x[1] = xpc[1 + 3 * ipt] - fj;
    x[2] = xpc[2 + 3 * ipt] - fk;
    // kernel weights 
    delta[ipt + m * npts] = deltaf(x) * weight;  
  }
}

void spread_col(double* Fec, const double* delta, const double* flc, 
                const int i0, const int ipt, const int npts, const int w3)
{
  #pragma omp simd aligned(Fec,delta: MEM_ALIGN)
  for (int i = 0; i < w3; ++i)
  {
    Fec[3 * (i + i0)] += delta[ipt + i * npts] * flc[3 * ipt]; 
    Fec[1 + 3 * (i + i0)] += delta[ipt + i * npts] * flc[1 + 3 * ipt]; 
    Fec[2 + 3 * (i + i0)] += delta[ipt + i * npts] * flc[2 + 3 * ipt]; 
  }
}

void interp_col(const double* Fec, const double* delta, double* flc, 
                const int i0, const int ipt, const int npts, const int w3)
{
  double flsum, glsum, hlsum; flsum = glsum = hlsum = 0;
  #pragma omp simd aligned(Fec,flc,delta: MEM_ALIGN) reduction(+:flsum,glsum,hlsum) 
  for (int i = 0; i < w3; ++i)
  {
    flsum += Fec[3 * (i + i0)] * delta[ipt + i * npts]; 
    glsum += Fec[1 + 3 * (i + i0)] * delta[ipt + i * npts]; 
    hlsum += Fec[2 + 3 * (i + i0)] * delta[ipt + i * npts]; 
  }
  flc[3 * ipt] += flsum; flc[1 + 3 * ipt] += glsum; flc[2 + 3 * ipt] += hlsum;
}

#endif


