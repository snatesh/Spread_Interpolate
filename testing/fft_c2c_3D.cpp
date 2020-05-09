#include<fftw3.h>
#include<math.h>
#include<omp.h>
#include<iostream>

// testing complex to complex forward and backward in-place transforms
// for 3D vector field stored row-major as (k,j,i,l), l = 0:2
// We also check OpenMP integration
int main(int argc, char* argv[])
{
  // initialize threads 
  if (!fftw_init_threads()) 
  { 
    std::cerr << "Could not initialize threads" << std::endl;
    exit(1);
  }
 
  for (int nthread = 1; nthread <= 6; ++nthread)
  {
    omp_set_num_threads(nthread);
    // set num threads to w/e used by openmp
    fftw_plan_with_nthreads(omp_get_max_threads());
    
    double *in_copy;
    fftw_complex *in, *out;
    fftw_plan pF, pB;
    fftw_iodim *dims, *howmany_dims; 
    unsigned int Nx = 128;
    unsigned int Ny = 128;
    unsigned int Nz = 128;
 
    // set up iodims - we store as (k,j,i,l), l = 0:2 
    dims = (fftw_iodim*) fftw_malloc(3 * sizeof(fftw_iodim));    
    if (!dims) {std::cerr << "alloc failed\n"; exit(1);}
    // we want to do 1 fft for the entire 3 x 3D array     
    howmany_dims = (fftw_iodim*) fftw_malloc(1 * sizeof(fftw_iodim));
    if (!howmany_dims) {std::cerr << "alloc failed\n"; exit(1);}
    // size of k
    dims[0].n = Nz;
    // stride for k
    dims[0].is = 3 * Nx * Ny;
    dims[0].os = 3 * Nx * Ny;
    // size of j
    dims[1].n = Ny;
    // stride for j
    dims[1].is = 3 * Nx;
    dims[1].os = 3 * Nx;
    // size of i
    dims[2].n = Nx;
    // stride for i
    dims[2].is = 3;
    dims[2].os = 3;
    
    // 3 component vec field
    howmany_dims[0].n = 3;
    // stride of 1 b/w each component (interleaved)
    howmany_dims[0].is = 1;
    howmany_dims[0].os = 1;

    in = fftw_alloc_complex(Nz * Ny * Nx * 3);
    if (!in) {std::cerr << "alloc failed\n"; exit(1);}
    in_copy = fftw_alloc_real(Nz * Ny * Nx * 3); 
    if (!in_copy) {std::cerr << "alloc failed\n"; exit(1);}
    // alias out to in
    out = (fftw_complex*) in; 


    // create plans for forward and backward transforms
    // MUST do this before populating in arrays
    pF = fftw_plan_guru_dft(3, dims, 1, howmany_dims, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!pF) {std::cerr << "fftw forward planning failed\n"; exit(1);}
    pB = fftw_plan_guru_dft(3, dims, 1, howmany_dims, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (!pB) {std::cerr << "fftw backward planning failed\n"; exit(1);}
    
    // initialize input and make a copy for later comparison
    for (unsigned int i = 0; i < Nx * Ny * Nz * 3; ++i)
    {
      in[i][0] = drand48(); in[i][1] = 0;
      in_copy[i] = in[i][0];
    }

    // execute forward transform
    double time = omp_get_wtime(); 
    fftw_execute(pF);

    // execute backward transform
    fftw_execute(pB);
    std::cout << "Elapsed time = " << omp_get_wtime() - time << std::endl;
    
    // make sure result is the same as initial input
    double maxerr,err; maxerr = 0;
    unsigned int N = Nx * Ny * Nz;
    for (unsigned int i = 0; i < Nx * Ny * Nz * 3; ++i)
    {
      // compare difference (note we normalize by N)
      err = fabs(in[i][0]/N - in_copy[i]);
      maxerr = (maxerr >= err ? maxerr : err);
    }
    std::cout << "Max error = " << maxerr << std::endl;
 
    // destroy plans 
    fftw_destroy_plan(pF);
    fftw_destroy_plan(pB);
    // free memory
    fftw_free(in);
    fftw_free(in_copy);
    fftw_free(dims);
    fftw_free(howmany_dims);
  }
  return 0;
}
