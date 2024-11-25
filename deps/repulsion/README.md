# Repulsion
Fast tangent-point energy and preconditioners



## Installation:

The installation process is easy. Just clone the repo … to some place where your compiler can find the include directory. Or put the include directory onto your system’s PATH.

In your program, just use

    #include "Repulsion.h"

## Compiling + Linking:

Probably clang is the best compiler to use here. (We had some trouble with gcc, but I don't recall what exactly it was; since some dependencies wereremoved, gcc might work now, too). 
Make sure that the following dependencies are installed and found; they are all part of the Intel oneAPI which you might have to install first:

- the Intel MKL (nowadays called Intel oneMKL and part of the Intel oneAPI)
- the Intel Threading Building Blocks
- Intel OpenMP (libiomp5)

Moreover we need the following dependency:

- boost



For 64-bit integers, the link line should be something like

     -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

and the compiler options should be

     -DMKL_ILP64  -m64  -I"${MKLROOT}/include“

But it is also possible to use 32-bit integers. The MKL defines a macro MKL_INT for the integer type it uses (which in the project is typedefed to "mint" for "machine integer"). The currently used floating point type is double (typedefed to "mreal" for "machine real").

For further details on compiler and linker options, please refer to https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html. Make sure to select OpenMP as threading layer and to select libiomp5 (not libgomp) as OpenMP library.

The required environment variables can be set by running the script setvars.sh which is found in the oneAPI directory (e.g., on my macos system it is here: /opt/intel/oneapi/setvars.sh). But you can of course give MKLROOT also manually; on my system it is /opt/intel/oneapi/mkl/latest. Dunno where oneAPI installs itself on Linux.


