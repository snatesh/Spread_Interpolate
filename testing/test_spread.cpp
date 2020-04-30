#include<math.h>
#include<omp.h>
#include<iostream>
#include<iomanip>
#include<fstream>
#include"spread_interp.h"
#include"init.h"
#include"io.h"
#include"utils.h"

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

  // spreading width, num uniform pts on each axis, num particles
  const unsigned short w = 6, N = w * ((int) 64 / w); 
  // grid spacing, effective radius, num total columns
  const double h = 1; const unsigned int N2 = N * N;
  
  // num particles
  const unsigned int Np = 10; 

  // particle positions (x1,y1,z1,x2,y2,z2,...)
  double* xp = (double*) aligned_malloc(Np * sizeof(double));
  double* yp = (double*) aligned_malloc(Np * sizeof(double));
  double* zp = (double*) aligned_malloc(Np * sizeof(double));

  // lagrangian force density (f1,g1,h1,f2,g2,h2,...)
  double* fl = (double*) aligned_malloc(Np * sizeof(double));
  double* gl = (double*) aligned_malloc(Np * sizeof(double));
  double* hl = (double*) aligned_malloc(Np * sizeof(double));

  // Eulerian force density array (F1,G1,H1,F2,G2,H2,...)
  double* Fe = (double*) aligned_malloc(N2 * N * sizeof(double));
  double* Ge = (double*) aligned_malloc(N2 * N * sizeof(double));
  double* He = (double*) aligned_malloc(N2 * N * sizeof(double));

  // firstn(i,j) holds index of first particle in column(i,j)
  int* firstn = (int*) aligned_malloc(N2 * sizeof(int));

  // number(i,j) holds number of partices in column(i,j)
  unsigned int* number = (unsigned int*) aligned_malloc(N2 * sizeof(unsigned int));

  // nextn(i) to hold index of the next particle in the column with particle i
  int* nextn = (int*) aligned_malloc(Np * sizeof(int));
  
  init(Np, N, h, xp, yp, zp, fl, gl, hl, Fe, Ge, He, firstn, nextn, number);
  
  const bool write = true;
  
  if (write) write_to_file(xp, yp, zp, Np, "particles.txt");
      
  spread_interp(xp, yp, zp, fl, gl, hl, Fe, Ge, He, firstn, nextn, number, w, h, N, true);
  
  if (write)
  { 
    write_to_file(Fe, Ge, He, N2 * N, "spread.txt"); 
    write_coords(N,h,"coords.txt");
    write_to_file(fl, gl, hl, Np, "forces.txt");
  }
      
  // reinitialize force for interp
  #pragma omp parallel for
  for (unsigned int i = 0; i < Np; ++i) { fl[i] = gl[i] = hl[i] = 0;}
  
  spread_interp(xp, yp, zp, fl, gl, hl, Fe, Ge, He, firstn, nextn, number, w, h, N, false);
  
  if (write) write_to_file(fl, gl, hl, Np, "interp.txt"); 
  
  aligned_free(xp);
  aligned_free(yp);
  aligned_free(zp);
  aligned_free(fl);
  aligned_free(gl);
  aligned_free(hl);
  aligned_free(firstn);
  aligned_free(nextn);
  aligned_free(number);
  aligned_free(Fe);
  aligned_free(Ge);
  aligned_free(He);
  return 0;
}

 

