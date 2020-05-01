#include<iostream>
#include<iomanip>
#include<fstream>
#include<sstream>
#include<string>
#include"io.h"

using std::setw;
using std::setprecision;

void write_to_file(const double* F, const unsigned int N, const char* fname)
{
  std::ofstream file; file.open(fname);
  if (file.is_open())
  {
    for (unsigned int i = 0; i < N; ++i)
    {
      for (unsigned int j = 0; j < 3; ++j)
      {
        file << setprecision(16) << F[j + i * 3] << " ";
      }
      file << std::endl;
    }
    file.close();
  }
  else
  {
    std::cout << "Unable to open file " << fname << std::endl;
    exit(1);
  }
}   

void read_from_file(double* F, const char* fname)
{
  std::ifstream file(fname);
  if (file.is_open())
  {
    std::string line;
    unsigned int i = 0;
    while (getline(file,line))
    {
      std::stringstream ss(line);
      ss >> F[i * 3] >> F[1 + i * 3] >> F[2 + i * 3];
      i += 1;
    }
    file.close();
  }
  else
  {
    std::cout << "Unable to open file " << fname << std::endl;
    exit(1);
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
    exit(1);
  }
} 
