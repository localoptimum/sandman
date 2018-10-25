# SANDMAN-LIB INSTALLATION INSTRUCTIONS

## Prerequisites

* NVIDIA GPU - go buy one
* NVIDIA CUDA - download from nvidia's website
  (https://developer.nvidia.com/cuda-zone at the time of writing)
* NVIDIA CUDA prerequisites (install them all)
* cmake
* CUDA-compatible compiler environment (build some cuda examples and
  fix if necessary)
* python3, matplotlib for the plotting tool script

## Steps to Install

1. Make a build directory
2. cd into the build directory
3. cmake ../
4. make
5. sudo make install

That will put the library and include files into your system tree, by
default these are:

* /usr/lib for the library
* /usr/include for the header
* /usr/bin for the graph visualisation script "sandmanPlot.py"

