#include<chebyshev.h>
#include"utils.h"
#include<iostream>
#include<iomanip>
int main(int argc, char* argv[])
{
  unsigned int N = atoi(argv[1]);
  double* cpts = (double*) aligned_malloc(N * sizeof(double));
  double* cwts = (double*) aligned_malloc(N * sizeof(double)); 
  double a,b; a = -1; b = 1;
  
  clencurt(cpts, cwts, a, b, N);
  
  std::cout << "Points:\n";
  for (unsigned int i = 0; i < N; ++i)
  {
    std::cout << std::setprecision(16) << cpts[i] << std::endl;
  }

  std::cout << "Weights:\n";
  for (unsigned int i = 0; i < N; ++i)
  {
    std::cout << std::setprecision(16) << cwts[i] << std::endl;
  } 

  aligned_free(cpts);
  aligned_free(cwts);
  return 0;
}
