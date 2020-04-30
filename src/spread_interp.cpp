#include<math.h>
#include<omp.h>
#include<stdalign.h>
#include"spread_interp.h"


#ifdef __SSE2__
  #define MEM_ALIGN 16
#elif __AVX__
  #define MEM_ALIGN 32
#elif defined(__MIC__)
  #define MEM_ALIGN 64
#endif

void get_delta_col(double* delta, const double* xpc, const double* ypc, const double* zpc, 
                    const double fi, const double fj, const double weight, const int m, 
                    const int k, const int npts, const int w, const int h)
{
  #pragma omp simd aligned(delta,xpc,ypc,zpc: MEM_ALIGN) // vectorization over particles in col
  for (unsigned int ipt = 0; ipt < npts; ++ipt) 
  {
    // unwrapped z coordinates
    double fk = (double) (((int)zpc[ipt] / h) + k - w/2 + 1) * h;
    alignas(MEM_ALIGN) double x[3]; 
    x[0] = xpc[ipt] - fi;
    x[1] = ypc[ipt] - fj;
    x[2] = zpc[ipt] - fk;
    // kernel weights 
    delta[ipt + m * npts] = deltaf(x) * weight;  
  }
} 

void spread_col(double* Fec, double* Gec, double* Hec, const double* delta, 
                const double* flc, const double* glc, const double* hlc,
                const int i0, const int ipt, const int npts, const int w3)
{
  #pragma omp simd aligned(Fec,Gec,Hec,flc,glc,hlc,delta: MEM_ALIGN)// vectorize over eulerian pts
  for (int i = 0; i < w3; ++i)
  {
    Fec[i + i0] += delta[ipt + i * npts] * flc[ipt]; 
    Gec[i + i0] += delta[ipt + i * npts] * glc[ipt]; 
    Hec[i + i0] += delta[ipt + i * npts] * hlc[ipt]; 
  }
}

void interp_col(const double* Fec, const double* Gec, const double* Hec, 
                const double* delta, double* flc, double* glc, double* hlc,
                const int i0, const int ipt, const int npts, const int w3)
{
  double flsum, glsum, hlsum; flsum = glsum = hlsum = 0;
  #pragma omp simd aligned(Fec,Gec,Hec,flc,glc,hlc,delta: MEM_ALIGN) reduction(+:flsum,glsum,hlsum)// vectorize over lagrangian pts 
  for (int i = 0; i < w3; ++i)
  {
    flsum += Fec[i + i0] * delta[ipt + i * npts]; 
    glsum += Gec[i + i0] * delta[ipt + i * npts]; 
    hlsum += Hec[i + i0] * delta[ipt + i * npts]; 
  }
  flc[ipt] = flsum; glc[ipt] = glsum; hlc[ipt] = hlsum;
}

void spread_interp(double* xp, double* yp, double* zp, double* fl, double* gl, 
                   double* hl, double* Fe, double* Ge, double* He, int* firstn, 
                   int* nextn, unsigned int* number, const unsigned short w, 
                   const double h, const unsigned short N, const bool mode)
{

  const double weight = (mode ? 1 : h * h * h);
  const unsigned short w2 = w * w; const unsigned int N2 = N * N; 
  const unsigned short w3 = w2 * w;
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
          alignas(MEM_ALIGN) double Fec[w2 * N];  
          alignas(MEM_ALIGN) double Gec[w2 * N];  
          alignas(MEM_ALIGN) double Hec[w2 * N];  
          gather(w2 * N, Fec, Fe, indc3D);
          gather(w2 * N, Gec, Ge, indc3D);
          gather(w2 * N, Hec, He, indc3D);
          
          // number of pts in this column, particle indices
          unsigned int npts = number[jj + ii * N], indx[npts];
          for (unsigned int ipt = 0; ipt < npts; ++ipt) {indx[ipt] = l; l = nextn[l];}
          
          // gather lagrangian pts and forces into xpc and flc
          alignas(MEM_ALIGN) double xpc[npts]; 
          alignas(MEM_ALIGN) double ypc[npts]; 
          alignas(MEM_ALIGN) double zpc[npts]; 
          alignas(MEM_ALIGN) double flc[npts]; 
          alignas(MEM_ALIGN) double glc[npts]; 
          alignas(MEM_ALIGN) double hlc[npts]; 
          gather(npts, xpc, xp, indx);
          gather(npts, ypc, yp, indx);
          gather(npts, zpc, zp, indx);
          gather(npts, flc, fl, indx);
          gather(npts, glc, gl, indx);
          gather(npts, hlc, hl, indx);
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
                get_delta_col(delta, xpc, ypc,  zpc, fi, fj, weight, m, 
                               k, npts, w, h);
  
                ////#pragma omp simd aligned(delta,xpc,ypc,zpc: MEM_ALIGN) // vectorization over particles in col
                //for (unsigned int ipt = 0; ipt < npts; ++ipt) 
                //{
                //  // unwrapped z coordinates
                //  double fk = (double) (((int)zpc[ipt] / h) + k - w/2 + 1) * h;
                //  // kernel weights 
                //  delta[ipt + m * npts] = deltaf(xpc[ipt] - fi, \
                //                                 ypc[ipt] - fj, \
                //                                 zpc[ipt] - fk) * weight;  

                //}
              }
            }
          }
  
          if (mode)
          {         
            // update Eulerian density
            for (unsigned int ipt = 0; ipt < npts; ++ipt)
            {
              int i0 = w2 * ((int) zpc[ipt] / h - w/2 + 1);
              
              spread_col(Fec, Gec, Hec, delta, flc, glc, hlc,
                         i0, ipt, npts, w);
              ////#pragma omp simd aligned(Fec,Gec,Hec,flc,glc,hlc,delta: MEM_ALIGN)// vectorize over eulerian pts
              //for (int i = 0; i < w2 * w; ++i)
              //{
              //  Fec[i + i0] += delta[ipt + i * npts] * flc[ipt]; 
              //  Gec[i + i0] += delta[ipt + i * npts] * glc[ipt]; 
              //  Hec[i + i0] += delta[ipt + i * npts] * hlc[ipt]; 
              //}
            }
            // scatter back to global eulerian grid
            scatter(w2 * N, Fec, Fe, indc3D);
            scatter(w2 * N, Gec, Ge, indc3D);
            scatter(w2 * N, Hec, He, indc3D);
          } 
          else
          {
            // interpolate lagrangian density
            for (unsigned int ipt = 0; ipt < npts; ++ipt)
            {
              int i0 = w2 * ((int) zpc[ipt] / h - w/2 + 1);
              interp_col(Fec, Gec, Hec, delta, flc, glc, hlc,
                         i0, ipt, npts, w);
              
              ////#pragma omp simd aligned(Fec,Gec,Hec,flc,glc,hlc,delta: MEM_ALIGN) // vectorize over lagrangian pts 
              //for (int i = 0; i < w2 * w; ++i)
              //{
              //  flc[ipt] += Fec[i + i0] * delta[ipt + i * npts]; 
              //  glc[ipt] += Gec[i + i0] * delta[ipt + i * npts]; 
              //  hlc[ipt] += Hec[i + i0] * delta[ipt + i * npts]; 
              //}
            }   
            // scatter back to global lagrangian grid
            scatter(npts, flc, fl, indx);
            scatter(npts, glc, gl, indx);
            scatter(npts, hlc, hl, indx);
          }
        } 
      // go to next column in group
      } 
    // go to next group
    }
  }
}    


