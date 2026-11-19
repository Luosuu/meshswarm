#!/bin/bash
# Builds ./amr. Change this as you like; the grader runs it as `bash build.sh`.
set -e
${CC:-gcc} -O3 -std=gnu99 -march=native -fopenmp -DPRKVERSION='"2.17"' -DDOUBLE=1 -DRADIUS=2 -DSTAR=1 -DLOOPGEN=0 -DVERBOSE=0 -DRESTRICT_KEYWORD=1 -I. amr.c wtime.c -lm -o amr
