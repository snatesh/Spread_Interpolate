#include<iostream>
#include<iomanip>
#include<fstream>
#include"io.h"

using std::setw;
using std::setprecision;

void write_to_file(const double* F, const double* G, const double* H,
                   const unsigned int N, const char* fname)
{
  std::ofstream file; file.open(fname);
  if (file.is_open())
  {
    for (unsigned int i = 0; i < N; ++i)
    {
      file << setprecision(16) << F[i] << " "
           << setprecision(16) << G[i] << " "
           << setprecision(16) << H[i] << "\n";
    }
    file.close();
  }
  else
  {
    std::cout << "Unable to open file " << fname << std::endl;
  }
}   

void write_coords(const unsigned int N, const double h, const char* fname)
{
  std::ofstream file; file.open(fname);
  if (file.is_open())
  {
    for (unsigned int k = 0; k < N; ++k)
    {
      for (unsigned int j = 0; j < N; ++j)
      {
        for (unsigned int i = 0; i < N; ++i)
        {
          file << h * i << " " << h * j << " " << h * k << std::endl;
        }
      }
    }
    file.close();
  }
  else
  {
    std::cout << "Unable to open file " << fname << std::endl;
  }
} 
