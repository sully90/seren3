#!/bin/sh
#$ -N mpijob
###$ -M ds381@sussex.ac.uk
#$ -m bea
#$ -cwd
#$ -pe openmpi 8
#$ -q mps.q
#$ -S /bin/bash
# source modules environment:
#module load gcc/4.8.1
#module load openmpi/gcc/1.8.1

#module load intel/parallel_studio_xe/2014/14.0.1
#module load mvapich2/intel/64/1.9

module load gcc/4.8.1

ulimit -l unlimited
which mpirun

echo $@

mpirun -np $NSLOTS $@
