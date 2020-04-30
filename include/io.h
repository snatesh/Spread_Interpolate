#ifndef IO_H
#define IO_H

void write_to_file(const double* F, const double* G, const double* H,
                   const unsigned int N, const char* fname);
void write_coords(const unsigned int N, const double h, const char* fname);

#endif
