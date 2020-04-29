# Spread_Interpolate
Shared memory parallel implementation to spread Lagrangian sources at non-uniform points onto an Eulerian grid, and interpolate 
function values on the Eulerian grid to Lagrangian points using compactly supported kernels.


### Build Dependencies ###
You will need to `apt install` at least the following dependencies:

* build-essential
* cmake
* libomp-dev
* gcc 7.5.0 or later 

### Build Instructions ###
Now, execute the following from the top of the source tree: 
```
$ mkdir build && cd build
$ cmake ..
$ make -j6 (or however many threads you'd like to use)
$ make install
```
Executing the commands above will build all libraries and executables. The libraries are
installed in `$INSTALL_PATH/lib`. Executables are installed in `$INSTALL_PATH/bin`. 
By default, `$INSTALL_PATH` is the top of the source tree.

Currently, the only executable is a benchmark test to measure speedup over serial performance.


