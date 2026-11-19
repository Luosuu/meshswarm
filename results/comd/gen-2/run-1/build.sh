#!/bin/bash
# Builds ./CoMD. Change this as you like; the grader runs it as `bash build.sh`.
set -e
${CC:-gcc} -std=gnu99 -O3 -march=native -fno-math-errno -fopenmp -DDOUBLE -I. *.c -lm -o CoMD
