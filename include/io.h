#ifndef IO_H
#define IO_H

void write_to_file(const double* F, const unsigned int N, const char* fname);
void write_coords(const unsigned int N, const double h, const char* fname);
void read_from_file(double* F, const char* fname);

#endif
