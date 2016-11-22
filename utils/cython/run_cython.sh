#!/bin/bash

x=$1
fname=(`basename ${x%%.*}`)
echo $fname

cython -a $fname.pyx && icc -shared -fPIC -O3 $fname.c -o $fname.so -I/lustre/scratch/astro/ds381/yt-x86_64/include/python2.7/
