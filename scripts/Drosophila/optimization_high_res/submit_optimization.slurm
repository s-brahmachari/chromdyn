#!/bin/bash -l

#SBATCH --job-name=opt
#SBATCH --account=sb95
#SBATCH --partition=highmem
#SBATCH --nodes=1            # this can be more, up to 22 on aries
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=256G
#SBATCH --gres=gpu:0
#SBATCH --time=6:00:00
#SBATCH --export=ALL

module purge
module load foss/2020b Launcher_GPU OpenMPI 

source $HOME/miniforge3/bin/activate
conda activate openmm-env

# Controling Launcher and showing some job info
export LAUNCHER_WORKDIR=`pwd`
export LAUNCHER_JOB_FILE=$PWD/launcher_jobs_sim
export LAUNCHER_BIND=1

iteration=$1
dense=$2
eta=$3

#run optimization step (this is fast)
python run_optimization.py ${iteration} input/ ${dense} ${eta} output_${iteration}/run_* >output_${iteration}/optmin.log
date

iteration=$(($iteration+1))

if [ "$iteration" -lt 60 ]; then
    sbatch submit_simulation.slurm ${iteration} ${dense} ${eta}
fi





