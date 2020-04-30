#ifndef _SPREAD_INTERP_H
#define _SPREAD_INTERP_H

#pragma omp declare simd
inline double const deltaf(const double x, const double y, const double z);


// flattened index into 3D array
inline unsigned int const at(unsigned int i, unsigned int j,unsigned int k,\
                             const unsigned int Nx, const unsigned int Ny);


// flatted index into 3D array, corrected for periodic boundaries
inline unsigned int const pbat(int i, int j, int k, const int Nx, const int Ny);


// flatted index into 3D array, corrected for periodic boundaries
inline unsigned int const pbat(int i, int j, int k, const int Nx, const int Ny, const int Nz);



// gather data from src at inds into trg
inline void gather(unsigned int N, double* trg, double const* src, const unsigned int* inds);

// scatter data from trg at inds into src

inline void scatter(unsigned int N, double const* trg, double* src, const unsigned int* inds);

// spreading (mode=true) and interpolation (mode=false)
void spread_interp(double* xp, double* yp, double* zp, double* fl, double* gl, 
                   double* hl, double* Fe, double* Ge, double* He, int* firstn, 
                   int* nextn, unsigned int* number, const unsigned short w, 
                   const double h, const unsigned short N, const bool mode);
#endif
