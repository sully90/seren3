#!/bin/sh
#$ -N fbaryon
###$ -M ds381@sussex.ac.uk
#$ -m bea
#$ -cwd
#$ -pe openmpi 16
#$ -q mps.q
#$ -S /bin/bash
# source modules environment:
#module load gcc/4.8.1
#module load openmpi/gcc/1.8.1

#module load intel/parallel_studio_xe/2014/14.0.1
#module load mvapich2/intel/64/1.9

source /lustre/scratch/astro/ds381/yt-x86_64/bin/activate

module load gcc/4.8.1
module load openmpi/gcc/64/1.7.3
module unload mvapich2/intel/64/1.9

export PATH=${HOME}/apps/bin:${PATH}

ulimit -l unlimited
which mpirun
which python
which ramses2gadget
which AHF-v1.0-084

#mpirun --mca btl ^openib -np $NSLOTS /home/d/ds/ds381/Code/ramses-rt-metals/trunk/ramses/bin/ramses3d nml.nml > run.log

#INBASE=$1
#INBASE=/lustre/scratch/astro/ds381/simulations/bpass/ramses_sed_bpass/rbubble_200
INBASE=/lustre/scratch/astro/ds381/simulations/bpass/ramses/
OUTPUTS=$INBASE/output_*

for OUT in $OUTPUTS; do
	IOUT=${OUT:(-5)}
	echo $IOUT
	module load mvapich2/intel/64/1.9
	python ~/seren3/scripts/run_ahf.py $INBASE $IOUT $NSLOTS
	module unload mvapich2/intel/64/1.9
	mpirun -np ${NSLOTS} python ~/seren3/scripts/mpi/fbaryon.py $INBASE $IOUT
done
