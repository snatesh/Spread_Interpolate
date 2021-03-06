#include"spread_interp.h"
#include"init.h"
#include"io.h"
#include"utils.h"
#include<math.h>
#include<omp.h>
#include<iostream>
#include<iomanip>
#include<fstream>

using std::setw;
using std::setprecision;

/*
  NOTES:
    - if w even, all particles in a column interact 
      with the SAME subarray of fluid grid
    - if w odd, particles left/right of column center interact 
      with different, 1-shifted subarrays, so columns should not be aligned with grid pts
    - though there are (N-1)^2 columns on the grid, we use N^2, where the additional 2n-1
      columns fill the ghost region at the upper bndry of the periodic axes.
*/

int main(int argc, char* argv[])
{
  const unsigned int nreps = 1, maxthreads = 50; 
  double Times_interp_serial[maxthreads];
  double Times_spread_serial[maxthreads];
  double Times_interp[maxthreads];
  double Times_spread[maxthreads];
  unsigned int w = 6; 
  unsigned int Nbase = w;
  // first get serial times for increasing problem size
  for (unsigned int ithread = 1; ithread <= maxthreads; ++ithread)
  {
    // num uniform pts on each axis, num particles
    unsigned int N = Nbase * ithread, Nwrap = N; 
    // grid spacing, effective radius, num total columns
    const double h = 1, Rh = 1.7305 * h, L = h * N; 
    // correct N for pbc
    bool pbc = true;
    if (pbc) N += w;  
    const unsigned int N2 = N * N;
    
    // max packing density
    double phimax = 0.5;
    // num particles
    const unsigned int Np = (int) (3.0 / 4.0 / M_PI * phimax * pow(L / Rh, 3));
    std::cout << "Spreading and interpolating with " << Np << " particles\n";
    // particle positions (x1,y1,z1,x2,y2,z2,...)
    double* xp = (double*) aligned_malloc(Np * 3 * sizeof(double));

    // lagrangian force density (f1,g1,h1,f2,g2,h2,...)
    double* fl = (double*) aligned_malloc(Np * 3 * sizeof(double));

    // Eulerian force density array (F1,G1,H1,F2,G2,H2,...)
    double *Fe = (double*) aligned_malloc(N2 * N * 3 * sizeof(double)), *Fe_wrap;
    if (pbc) 
    {
      Fe_wrap = (double*) aligned_malloc(Nwrap * Nwrap * Nwrap * 3 * sizeof(double));
    }

    // firstn(i,j) holds index of first particle in column(i,j)
    int* firstn = (int*) aligned_malloc(N2 * sizeof(int));

    // number(i,j) holds number of partices in column(i,j)
    unsigned int* number = (unsigned int*) aligned_malloc(N2 * sizeof(unsigned int));

    // nextn(i) to hold index of the next particle in the column with particle i
    int* nextn = (int*) aligned_malloc(Np * sizeof(int));

    omp_set_num_threads(1);
    double times_spread = 0, times_interp = 0;
    for (unsigned int irep = 0; irep < nreps; ++irep)
    {
      if (!pbc) init(Np, N, h, xp, fl, Fe, firstn, nextn, number);
      else init(Np, N, w, h, xp, fl, Fe, firstn, nextn, number);  
      
      double time;
      if (!pbc) 
      {
        time = omp_get_wtime();
        spread_interp(xp, fl, Fe, firstn, nextn, number, w, h, N, true);
        times_spread += omp_get_wtime() - time;
      }
      else 
      {
        time = omp_get_wtime();
        spread_interp_pbc(xp, fl, Fe, Fe_wrap, firstn, nextn, number, w, h, N, true);
        times_spread += omp_get_wtime() - time;
      }
      Times_spread_serial[ithread-1] = times_spread/((double) nreps);
      
      // reinitialize force for interp
      #pragma omp parallel for
      for (unsigned int i = 0; i < Np * 3; ++i) fl[i] = 0;
      if (pbc)
      {
        #pragma omp parallel for
        for (unsigned int i = 0; i < N2 * N * 3; ++i) {Fe[i] = 0;}
      }

      if (!pbc) 
      {
        time = omp_get_wtime();
        spread_interp(xp, fl, Fe, firstn, nextn, number, w, h, N, false);
        times_interp += omp_get_wtime() - time;
      }
      else  
      {
        time = omp_get_wtime();
        spread_interp_pbc(xp, fl, Fe, Fe_wrap, firstn, nextn, number, w, h, N, false);
        times_interp += omp_get_wtime() - time;      
      }
    }
    Times_interp_serial[ithread-1] = times_interp/((double) nreps);
    std::cout << N << " " << Times_spread_serial[ithread-1] 
              << " " << Times_interp_serial[ithread-1] << std::endl;
    aligned_free(xp);
    aligned_free(fl);
    aligned_free(firstn);
    aligned_free(nextn);
    aligned_free(number);
    aligned_free(Fe);
	  if (pbc) aligned_free(Fe_wrap);
  }
  std::ofstream file_spread; file_spread.open("Times_spread_serial.txt");
  std::ofstream file_interp; file_interp.open("Times_interp_serial.txt");
  if (file_spread.is_open() && file_interp.is_open())
  {
    for (unsigned int ithread = 1; ithread <= maxthreads; ++ithread)
    {
      file_spread << setprecision(16) << Times_spread_serial[ithread-1] << std::endl;
      file_interp << setprecision(16) << Times_interp_serial[ithread-1] << std::endl;
    }
    file_spread.close();
    file_interp.close();
  }
  else
  {
    std::cout << "Unable to open file" << std::endl;
  }

  // now increase proc and problem size
  for (unsigned int ithread = 1; ithread <= maxthreads; ++ithread)
  {
    // num uniform pts on each axis, num particles
    unsigned int N = Nbase * ithread, Nwrap = N; 
    // grid spacing, effective radius, num total columns
    const double h = 1, Rh = 1.7305 * h, L = h * N; 
    // correct N for pbc
    bool pbc = true;
    if (pbc) N += w;  
    const unsigned int N2 = N * N;
    
    // max packing density
    double phimax = 0.5;
    // num particles
    const unsigned int Np = (int) (3.0 / 4.0 / M_PI * phimax * pow(L / Rh, 3));
    std::cout << "Spreading and interpolating with " << Np << " particles\n";
    // particle positions (x1,y1,z1,x2,y2,z2,...)
    double* xp = (double*) aligned_malloc(Np * 3 * sizeof(double));

    // lagrangian force density (f1,g1,h1,f2,g2,h2,...)
    double* fl = (double*) aligned_malloc(Np * 3 * sizeof(double));

    // Eulerian force density array (F1,G1,H1,F2,G2,H2,...)
    double *Fe = (double*) aligned_malloc(N2 * N * 3 * sizeof(double)), *Fe_wrap;
    if (pbc) 
    {
      Fe_wrap = (double*) aligned_malloc(Nwrap * Nwrap * Nwrap * 3 * sizeof(double));
    }

    // firstn(i,j) holds index of first particle in column(i,j)
    int* firstn = (int*) aligned_malloc(N2 * sizeof(int));

    // number(i,j) holds number of partices in column(i,j)
    unsigned int* number = (unsigned int*) aligned_malloc(N2 * sizeof(unsigned int));

    // nextn(i) to hold index of the next particle in the column with particle i
    int* nextn = (int*) aligned_malloc(Np * sizeof(int));

    omp_set_num_threads(ithread);
    double times_spread = 0, times_interp = 0;
    for (unsigned int irep = 0; irep < nreps; ++irep)
    {
      if (!pbc) init(Np, N, h, xp, fl, Fe, firstn, nextn, number);
      else init(Np, N, w, h, xp, fl, Fe, firstn, nextn, number);  
      
      double time;
      if (!pbc) 
      {
        time = omp_get_wtime();
        spread_interp(xp, fl, Fe, firstn, nextn, number, w, h, N, true);
        times_spread += omp_get_wtime() - time;
      }
      else 
      {
        time = omp_get_wtime();
        spread_interp_pbc(xp, fl, Fe, Fe_wrap, firstn, nextn, number, w, h, N, true);
        times_spread += omp_get_wtime() - time;
      }
      Times_spread[ithread-1] = times_spread/((double) nreps);
      
      // reinitialize force for interp
      #pragma omp parallel for
      for (unsigned int i = 0; i < Np * 3; ++i) fl[i] = 0;
      if (pbc)
      {
        #pragma omp parallel for
        for (unsigned int i = 0; i < N2 * N * 3; ++i) {Fe[i] = 0;}
      }

      if (!pbc) 
      {
        time = omp_get_wtime();
        spread_interp(xp, fl, Fe, firstn, nextn, number, w, h, N, false);
        times_interp += omp_get_wtime() - time;
      }
      else  
      {
        time = omp_get_wtime();
        spread_interp_pbc(xp, fl, Fe, Fe_wrap, firstn, nextn, number, w, h, N, false);
        times_interp += omp_get_wtime() - time;      
      }
    }
    Times_interp[ithread-1] = times_interp/((double) nreps);
    std::cout << ithread << " " << Times_spread[ithread-1] 
              << " " << Times_interp[ithread-1] << std::endl;
    aligned_free(xp);
    aligned_free(fl);
    aligned_free(firstn);
    aligned_free(nextn);
    aligned_free(number);
    aligned_free(Fe);
	  if (pbc) aligned_free(Fe_wrap);
  }
  file_spread.open("Times_spread.txt");
  file_interp.open("Times_interp.txt");
  if (file_spread.is_open() && file_interp.is_open())
  {
    for (unsigned int ithread = 1; ithread <= maxthreads; ++ithread)
    {
      file_spread << setprecision(16) << Times_spread[ithread-1] << std::endl;
      file_interp << setprecision(16) << Times_interp[ithread-1] << std::endl;
    }
    file_spread.close();
    file_interp.close();
  }
  else
  {
    std::cout << "Unable to open file" << std::endl;
  }
  return 0;
}

 

