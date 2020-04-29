CC=g++


all: spread_interp_bench 

spread_interp_bench: spread_interp_bench.cpp io.o init.o spread_interp.o
	$(CC) -fopenmp io.o init.o spread_interp.o spread_interp_bench.cpp -o spread_interp_bench

io.o: io.cpp
	$(CC) -O2 -ftree-vectorize -c io.cpp -o io.o

init.o: init.cpp
	$(CC) -O2 -ftree-vectorize -fopt-info -fopenmp -c init.cpp -o init.o

spread_interp.o: spread_interp.cpp
	$(CC) -O2 -ftree-vectorize -fopt-info -fopt-info-missed -mavx -fopenmp -c spread_interp.cpp -o spread_interp.o

clean:
	rm -f spread_interp_bench *.o *.out
