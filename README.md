# Repulsive Shells
This repository contains C++ code for our paper "[Repulsive Shells](https://dx.doi.org/10.1145/3658174)" published at 
SIGGRAPH 2024 by [Josua Sassen](https://josuasassen.com), 
[Henrik Schumacher](https://www.tu-chemnitz.de/mathematik/harmonische_analysis/schumacher/), 
[Martin Rumpf](https://ins.uni-bonn.de/staff/rumpf), and 
[Keenan Crane](https://www.cs.cmu.edu/~kmcrane/).
Specifically, it contains code to compute geodesics, the exponential map, and weighted geodesic means on the 
*Shape Space of Repulsive Shells*. Furthermore, data and configurations demonstrating these concepts on a set of 
examples are included as well.

The code is based on [GOAST 👻](https://gitlab.com/numod/goast) a C++ library for research in Geometry Processing 
focused on variational approaches. Thus, it inherits its dependencies and adds a few additional ones (see below).
If you should have any questions, feel free to reach out!

**NOTE:** So far, this code has only been tested on Linux, specifically Ubuntu, and macOS. Tests on further operating systems are 
planned, but until then, the instructions below might be incomplete. Again, if you should run into trouble, please 
reach out even if you have resolved the issues so we can adapt these instructions.

## Dependencies
We split the necessary dependencies for our code into ones that are included in this repository (directly or as git 
submodule) and ones that users needs to install on their own beforehand:

### Included Dependencies
 - [GOAST 👻](https://gitlab.com/numod/goast) (as git submodule)
 - an earlier version of [Repulsor](https://github.com/HenrikSchumacher/Repulsor) as efficient tangent-point implementation (directly)
 - [PQP](https://github.com/GammaUNC/PQP) for triangle-triangle distances  (directly)
 - [yaml-cpp](https://github.com/jbeder/yaml-cpp) for configuration files (git submodule)

### External Dependencies
 - [OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/)
 - [Eigen](https://eigen.tuxfamily.org/)
 - [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse)
 - [Boost](https://www.boost.org/)
 - [VTK](https://vtk.org/) is optional but provides more output 

## Building and Running
### Cloning / Git submodule
This repository includes git submodules, which means to clone the repository one has to use either

```bash
git clone https://github.com/jrsassen/repulsive-shells.git 
cd repulsive-shells # or which ever folder you cloned the project to
git submodule init # initialize submodules given in .gitmodules
git submodule update # clone submodules
```

or, more compactly,
```bash
git clone --recurse-submodules https://github.com/jrsassen/repulsive-shells.git
```
For a detailed explanation of git submodules, see for example the [git book](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

### Building
The project is compiled using the common CMake routine:
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```
Depending on how one installed the external dependencies this might not find all of them.
In this case, one needs to tell CMake where they are explicitly by setting the following options.

| Option                  | Explanation                                  |         Example value |
|-------------------------|----------------------------------------------|----------------------:|
| `SUITESPARSE_HOME`      | Directory of your SuiteSparse installation   |    `/opt/suitesparse` |
| `EIGEN3_INCLUDE_DIR`    | Directory of your Eigen installation         | `/usr/include/eigen3` |
| `OPENMESH_LIBRARY_DIR`  | Directory of your OpenMesh installation      |          `/usr/local` |
| `BOOST_ROOT`            | Directory of your Boost installation         |                       |

The following additional CMake options might also be interesting: 

| Option               | Explanation                                 |        Example value |
|----------------------|---------------------------------------------|---------------------:|
| `CMAKE_BUILD_TYPE`   | determine build mode                        | `Release` or `Debug` |
| `GOAST_WITH_VTK`     | whether to activate optional VTK dependency |        `ON` or `OFF` |
| `VTK_DIR`            | Directory of your VTK installation          |                      |
| `GOAST_WITH_OPENMP`  | whether to use OpenMP in the GOAST library  |        `ON` or `OFF` |
| `GOAST_WITH_MKL`     | whether to use MKL (primarily in Eigen)     |        `ON` or `OFF` |

Setting these options can be achieved by specifying them in the terminal in the following way
```bash
cmake -D<OPTION>=<VALUE> ..
```
or by using `cmake-gui`.

### Running
After building, a number of executables should be present in the `build` directory that can be run typically with
paths to YAML configuration files as commandline argument.
Here is a list of these executables, what they do, and if/which configuration examples are provided (in the `data` 
folder).

| Executable                      | Explanation                                                                                                 | Configurations                                                                    |
|---------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `AugmentedSurfaceInterpolation` | Compute geodesics (i.e. interpolations) on the space of repulsive shells                                    | `Interpolation_SCAPE_frontback.yaml` and `Interpolation_SphereThroughTunnel.yaml` |
| `AugmentedSurfaceExtrapolation` | Compute the exponential map (i.e. extrapolation) on the space of repulsive shells                           | `Extrapolation_Leaf.yaml` and `Extrapolation_HeadPunch.yaml`                      |
| `ElasticSurfaceExtrapolation`   | Compute the exponential map on the space of discrete shells (i.e. without tangent-point energy)             | `Extrapolation_Leaf.yaml` and `Extrapolation_HeadPunch.yaml`                      |
| `RepulsiveMeans`                | Compute weighted geodesic means on the space of repulsive shells                                            | `Mean_X.yaml`                                                                     |
| `ShapeFromMetric`               | Compute (almost) isometric immersions for a given metric                                                    | `SFM_HyperbolicDisk.yaml` and `SFM_Torus.yaml`                                    |
| `Prolongation`                  | Compute the spatial upsampling used in the paper. Path to meshes are given directly as arguments (see code) | ---                                                                               |

Executables take paths to YAML configuration files as commandline arguments. 
For example configurations, see the `data` folder. 
Please note, that the paths specified in these example configurations expect (a) that one runs the exectuables in the 
main directory of the repository, and (b) one creates a folder called `output` first.


## Tangent-point Energy Implementation
This code includes and uses two implementations of the discrete adaptive tangent-point energy proposed in our paper.

On the one hand, there is an implementation by 
[Henrik](https://www.tu-chemnitz.de/mathematik/harmonische_analysis/schumacher/), which is essentially an earlier 
version of the [Repulsor](https://github.com/HenrikSchumacher/Repulsor) library. This implementation focuses on 
efficiency and thus, e.g., exploits the vectorization capabilities of modern CPUs. In the code and configuration files,
it is dubbed `ScaryTPE` and it is the implementation that has been used for the examples shown in our paper and, 
generally, is the recommended one.

On the other hand, there is an implementation by [Josua](https://josuasassen.com) that is focused on readability over
performance and is mostly meant as additional illustration for the implementation notes and pseudocode provided in our
paper. Note that it differs from the former implementation in a few places, which might have a small impact on results. 
In the code and configuration files, it is dubbed `SpookyTPE`. As a small warning, even though focused on readability,
the code is not 100% cleaned up and could certainly be improved!

## License 
The code (except for the included dependencies) is provided under the MIT license (see the `LICENSE` file).
For the licenses of the included open source dependencies, please take a look at the corresponding subfolders in 
the `deps` folder.

## Citation
If you use this code or our work in general in a research paper, please remember to cite our paper! 
You can use the following bibtex:
```
@article{SaScRu24,
    author = {Sassen, Josua and Schumacher, Henrik and Rumpf, Martin and Crane, Keenan},
    title = {{Repulsive Shells}},
    year = {2024},
    issue_date = {July 2024},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {43},
    number = {4},
    issn = {0730-0301},
    doi = {10.1145/3658174},
    journal = {ACM Transaction on Graphics},
    articleno = {140},
    numpages = {22}
}
```
