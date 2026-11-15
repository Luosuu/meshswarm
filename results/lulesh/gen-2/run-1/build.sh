#!/bin/bash
# Builds ./lulesh. Change this as you like; the grader runs it as `bash build.sh`.
set -e
${CXX:-g++} -O3 -march=native -fno-math-errno -fopenmp -DUSE_MPI=0 -I. lulesh.cc lulesh-comm.cc lulesh-viz.cc lulesh-util.cc lulesh-init.cc -lm -o lulesh
