#include<math.h>
#include<stdio.h>
#include"utils.h"
#include<omp.h>

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
*/

int main()
{
  // spreading width, num groups of columns of 3D mesh
  unsigned short w = 6; unsigned short Ngroups = w * w;
  // grid spacing, num uniform pts on each axis, num particles
  double h = 1; unsigned short N,Nm1; unsigned int Np;
  N = 128; Nm1 = N-1; Np = 20000;
  // num total columns on 3D mesh
  unsigned int Ncols = Nm1 * Nm1; 
  // reserve space for particle positions
  double* xp = (double*) aligned_malloc(Np * sizeof(double));
  double* yp = (double*) aligned_malloc(Np * sizeof(double));
  double* zp = (double*) aligned_malloc(Np * sizeof(double));
  // reserve space for firstn(i,j) to hold index of first particle column(i,j)
  int* firstn = (int*) aligned_malloc(Ncols * sizeof(int));
  // reserve space for nextn(i) to hold index of 
  // the next particle in the column with particle i
  int* nextn = (int*) aligned_malloc(Np * sizeof(uint));
  for (int i = 0; i < Ncols; ++i) firstn[i] = -1;
  for (int i = 0; i < Np; ++i) nextn[i] = -1; 
  // populate particle positions and fill firstn and nextn (index-based linked list)
  int ii,jj,ind,indn;
  for (int i = 0; i < Np; ++i) 
  {
    // particles uniformly distributed in [0,h(N-1)]^3
    xp[i] = h * (N-1) * drand48(); ii = (int) xp[i] / h;
    yp[i] = h * (N-1) * drand48(); jj = (int) yp[i] / h;
    zp[i] = h * (N-1) * drand48(); ind = ii + jj * Nm1;
    if (firstn[ind] < 0) 
      firstn[ind] = i;
    else
    {
      indn = firstn[ind];
      while (nextn[indn] >= 0)
      {
        indn = nextn[indn];
      }
      nextn[indn] = i;
    }
  }
  
  // test the sorting of particles into columns done above
  for (int ii = 0; ii < Nm1; ++ii)
  {
    for (int jj = 0; jj < Nm1; ++jj)
    {
      ind = ii + jj * Nm1; indn = firstn[ind];
      if (indn >= 0)
      {
        printf("%lf\t%lf\t\n",xp[indn],yp[indn]);
        while (nextn[indn] > -1)
        {
          printf("%lf\t%lf\t\n",xp[nextn[indn]],yp[nextn[indn]]);
          indn = nextn[indn];
        }
        printf("\n");
      }
    }
  }
  
  aligned_free(xp);
  aligned_free(yp);
  aligned_free(zp);
  aligned_free(firstn);
  aligned_free(nextn);
}
