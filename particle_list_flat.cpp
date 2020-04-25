#include<math.h>
#include<stdio.h>
#include<cstdlib>
#include"utils.h"
#include<omp.h>
#include<iostream>
#include<fstream>

#define ALIGN  __attribute__((aligned (MEM_ALIGN)))

/*
  This script demonstrates the domain decomposition strategy we will use
  to turn the convolution problem between uniform and nonuniform points into an 
  embarassingly parallel one. First, we construct an array-based
  linked-list (links are indices) that tells us the index of the first particle
  in a column of the 3D mesh, and provides a way to iterate to the indices
  of the remaining particles in that column.

  We'll end up parallelizing over groups of columns that are well-separated w.r.t
  the kernel width, so no reductions/barriers are needed. This also enables
  auto-vectorization and implicit load balancing.
  
  NOTES:
    - if w even, all particles in a column interact 
      with the SAME subarray of fluid grid
    - if w odd, particles left/right of column center interact 
      with different, 1-shifted subarrays, so columns should not be aligned with grid pts
    - though there are (N-1)^2 columns on the grid, we use N^2, where the additional 2n-1
      columns fill the ghost region at the upper bndry of the periodic axes.
*/

// Normalized ES kernel for w = 6, beta/w = 1.7305, h =1

#pragma omp declare simd
inline double const deltaf(const double x, const double y, const double z)
{
  return exp(7.9602 * (sqrt(1 - x * x / 9) - 1)) * \
         exp(7.9602 * (sqrt(1 - y * y / 9) - 1)) * \
         exp(7.9602 * (sqrt(1 - z * z / 9) - 1)) / 8.031979529002255;
}

// flattened index into 3D array
inline unsigned int const at(unsigned int i, unsigned int j,unsigned int k,\
                             const unsigned int Nx, const unsigned int Ny)
{
  return i + Nx * (j + Ny * k);
}

// flatted index into 3D array, corrected for periodic boundaries
inline unsigned int const pbat(int i, int j, int k, const int Nx, const int Ny, const int Nz)
{
  i = (i >= 0 && i < Nx ? i : (i < 0 ? Nx + i : i - Nx)); 
  j = (j >= 0 && j < Ny ? j : (j < 0 ? Ny + j : j - Ny)); 
  k = (k >= 0 && k < Nz ? k : (k < 0 ? Nz + k : k - Nz)); 
  return i + Nx * (j + Ny * k);
}

// gather data from src at inds into trg
inline void gather(unsigned int N, double* trg, double const* src, const unsigned int* inds)
{
  for (unsigned int i = 0; i < N; ++i) 
  {
    for (unsigned int j = 0; j < 3; ++j)  
    {
      trg[j + 3 * i] = src[j + 3 * inds[i]];
    }
  }
}

// scatter data from trg at inds into src
inline void scatter(unsigned int N, double const* trg, double* src, const unsigned int* inds)
{
  for (unsigned int i = 0; i < N; ++i) 
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      src[j + 3 * inds[i]] = trg[j + 3 * i];
    }
  }
}

void write_to_file(const double* F, const unsigned int N, const char* fname)
{
  std::ofstream file; file.open(fname);
  for (unsigned int i = 0; i < N; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      file << F[j + i * 3] << " ";
    }
    file << std::endl;
  }
  file.close();
}   

int main()
{
  // spreading width, num groups of columns of 3D mesh,
  // num uniform pts on each axis, num particles
  const unsigned short w = 6, w2 = w * w, N = w * ((int) 128 / w), Np = 10;
  // grid spacing, num total columns on 3D mesh
  const double h = 1; const unsigned int N2 = N * N;
  // particle positions (x1,y1,z1,x2,y2,z2,...)
  double* xp = (double*) aligned_malloc(Np * 3 * sizeof(double));
  // lagrangian force density (f1,g1,h1,f2,g2,h2,...)
  double* fl = (double*) aligned_malloc(Np * 3 * sizeof(double));
  // Eulerian force density array (F1,G1,H1,F2,G2,H2,...)
  double* Fe = (double*) aligned_malloc(N2 * N * 3 * sizeof(double));
  // firstn(i,j) holds index of first particle in column(i,j)
  int* firstn = (int*) aligned_malloc(N2 * sizeof(int));
  // number(i,j) holds number of partices in column(i,j)
  unsigned int* number = (unsigned int*) aligned_malloc(N2 * sizeof(int));
  // nextn(i) to hold index of the next particle in the column with particle i
  int* nextn = (int*) aligned_malloc(Np * sizeof(int));
  #pragma omp parallel
  {
    #pragma omp for nowait
    for (unsigned int i = 0; i < Np; ++i) nextn[i] = -1; 
    #pragma omp for nowait
    for (unsigned int i = 0; i < N2; ++i) { firstn[i] = -1; number[i] = 0;}
    #pragma omp for
    for (unsigned int i = 0; i < N2 * N * 3; ++i) Fe[i] = 0;
  }
  // populate particle positions and fill firstn and nextn (no need to parallelize)
  int ii,jj,ind,indn;
  for (unsigned int i = 0; i < Np; ++i) 
  {
    // particles uniformly distributed in [2h,h(N-1)-2h]^3
    for (unsigned int j = 0; j < 3; ++j)
    {
      xp[j + 3 * i] =  drand48() * h * (N-1);//2 * h + (h * (N-1) - 4 * h) * drand48(); 
      fl[j + 3 * i] = 10;//2 * drand48() - 1;
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

  write_to_file(xp, Np, "particles.txt"); 
 
  // test the sorting of particles into columns done above
  //for (unsigned int ii = 0; ii < N; ++ii)
  //{
  //  for (unsigned int jj = 0; jj < N; ++jj)
  //  {
  //    ind = jj + ii * N; indn = firstn[ind];
  //    if (indn >= 0)
  //    {
  //      printf("N = %d\n",number[ind]);
  //      printf("%lf\t%lf\t%lf\n",xp[3 * indn],xp[1 + 3 * indn],xp[2 + 3 * indn]);
  //      while (nextn[indn] > -1)
  //      {
  //        printf("%lf\t%lf\t%lf\n",xp[3 * nextn[indn]],xp[1 + 3 * nextn[indn]],xp[2 + 3 * nextn[indn]]);
  //        indn = nextn[indn];
  //      }
  //      printf("\n");
  //    }
  //  }
  //}



  double time = omp_get_wtime(); 
  // loop over w^2 groups of columns
  for (unsigned int izero = 0; izero < w; ++izero)
  {
    for (unsigned int jzero = 0; jzero < w; ++jzero)
    {
      // parallelize over the N^2/w^2 columns in a group
      //#pragma omp parallel for //schedule(dynamic) 
      for (unsigned int ijcount = 0; ijcount < N2/w2; ++ijcount)
      {
        // column indices
        unsigned int jj = jzero + w * (ijcount / (N / w));
        unsigned int ii = izero + w * (ijcount % (N / w));

        // find first particle in column(ii,jj)
        int l = firstn[jj + ii * N];
        // if there is a particle, go on
        if (l >= 0 )
        {
          printf("Column %u\n",jj+ii*N);
          // global indices of w x w x N subarray influenced column(i,j)
          ALIGN unsigned int indc3D[w2*N];
          for (int i = 0; i < w; ++i) 
          {
            int i3D = ii + i - w/2 + 1;
            for (int j = 0; j < w; ++j)
            {
              int j3D = jj + j - w/2 + 1;
              for (int k3D = 0; k3D < N; ++k3D)
              {
                indc3D[at(i,j,k3D,w,w)] = pbat(i3D,j3D,k3D,N,N,N);
              }
            }
          }
          // gather eulerian foces for one column into contig mem
          ALIGN double Fec[w2 * N * 3];  
          gather(w2 * N, Fec, Fe, indc3D);
          // number of pts in this column, particle indices
          unsigned int npts = number[jj + ii * N], indx[npts];
          for (unsigned int i = 0; i < npts; ++i) {indx[i] = l; l = nextn[l];}
          // gather lagrangian pts and forces
          ALIGN double xpc[npts * 3]; 
          ALIGN double flc[npts * 3]; 
          gather(npts, xpc, xp, indx);
          gather(npts, flc, fl, indx);
          for (unsigned int i = 0; i < npts; ++i)
          {
            printf("%lf\t%lf\t%lf\n",xpc[3*i],xpc[1+3*i],xpc[2+3*i]);
          }
          printf("\n");
          // get the kernel w x w x w kernel weights for each particle in col 
          ALIGN double delta[w2 * w * npts];
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
                  delta[m + ipt * npts] = deltaf(xpc[3 * ipt] - fi, \
                                                 xpc[1 + 3 * ipt] - fj, \
                                                 xpc[2 + 3 * ipt] - fk);  
                  //printf("%lf\t%lf\n", fk, deltaf(xpc[3 * ipt] - fi, xpc[1 + 3 * ipt] - fj, xpc[2 + 3 * ipt] - fk));  
                }
              }
            }
          }
          // update Eulerian force density
          for (unsigned int ipt = 0; ipt < npts; ++ipt)
          {
            unsigned int mzero = w2 * ((int) xpc[2 + 3 * ipt] / h);
            //printf("%u\n", mzero);
            #pragma omp simd // vectorization over eulerian pts around particle
            for (unsigned int m = 0; m < w2 * w; ++m)
            {
              Fec[3 * m + mzero] += delta[m + ipt * npts] * flc[3 * ipt];
              Fec[1 + 3 * m + mzero] += delta[m + ipt * npts] * flc[1 + 3 * ipt];
              Fec[2 + 3 * m + mzero] += delta[m + ipt * npts] * flc[2 + 3 * ipt];
            }
          }
          // scatter back to global eulerian grid
          scatter(w2 * N, Fec, Fe, indc3D);
        }
      }
    }
  }
  std::cout << omp_get_wtime() -time << std::endl;
  write_to_file(Fe, N2 * N, "spread.txt"); 
  aligned_free(xp);
  aligned_free(fl);
  aligned_free(firstn);
  aligned_free(nextn);
  aligned_free(number);
  aligned_free(Fe);
}
