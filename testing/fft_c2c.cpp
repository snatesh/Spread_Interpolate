#include<fftw3.h>
#include<math.h>
#include<iostream>

// testing complex to complex forward and backward transforms

int main(int argc, char* argv[])
{
  fftw_complex *inF, *inB, *out;
  fftw_plan pF, pB;
  unsigned int N = 100;  

  // allocate with fftw_malloc - aligned to (usually) 16 byte bndry
  inF = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
  inB = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
  out = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
  
  // create plans for forward and backward transforms
  // MUST (rly should) do this before populating in arrays
  pF = fftw_plan_dft_1d(N, inF, out, FFTW_FORWARD, FFTW_ESTIMATE);
  pB = fftw_plan_dft_1d(N, out, inB, FFTW_BACKWARD, FFTW_ESTIMATE);
  
  for (unsigned int i = 0; i < N; ++i)
  {
    inF[i][0] = drand48();
    inF[i][1] = 0;
  }  

  // execute forward transform
  fftw_execute(pF);
  // execute backward transform (note result is multiplied by N)
  fftw_execute(pB);

  for (unsigned int i = 0; i < N; ++i)
  {
    // compare difference (note we normalize by N)
    std::cout << inF[i][0] - inB[i][0] / N << std::endl;
  }
 
  // destroy plans 
  fftw_destroy_plan(pF);
  fftw_destroy_plan(pB);
  // free memory
  fftw_free(inF);
  fftw_free(inB);
  fftw_free(out); 
  return 0;
}
