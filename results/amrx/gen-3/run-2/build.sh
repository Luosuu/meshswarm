#!/bin/bash
set -e
${CC:-gcc} -O3 -fopenmp -march=x86-64-v3 -std=gnu99 -DPRKVERSION='"2.17"' -DDOUBLE=1 -DRADIUS=2 -DSTAR=1 -DLOOPGEN=0 -DVERBOSE=0 -DRESTRICT_KEYWORD=1 -I. amr.c wtime.c -lm -o amr
