#!/bin/bash -l

#SBATCH --job-name=adamnull
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1            # this can be more, up to 22 on aries
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --export=ALL

module purge
module load foss/2020b Launcher_GPU OpenMPI 

source $HOME/miniforge3/bin/activate
conda activate openmm-env

max_replicas=$((SLURM_NNODES*8))
# Controling Launcher and showing some job info
export LAUNCHER_WORKDIR=`pwd`
export LAUNCHER_JOB_FILE=$PWD/launcher_jobs_sim
export LAUNCHER_BIND=1

iteration=$1
dense=$2
eta=$3

mkdir output_${iteration}

# Each iteration is an inversion
rm ${LAUNCHER_WORKDIR}/launcher_jobs_sim &> /dev/null
date
for i in `seq 1 $max_replicas` # SET to max number jobs
do
    echo "python run_simulation.py ${i} ${iteration} ${eta} > output_${iteration}/output_${i}.log" >> ${LAUNCHER_WORKDIR}/launcher_jobs_sim
done
$LAUNCHER_DIR/paramrun
wait
date

#run optimization step (this is fast)
python run_optimization.py ${iteration} input/ ${dense} ${eta} output_${iteration}/run_* >output_${iteration}/optmin.log
date

iteration=$(($iteration+1))

if [ "$iteration" -lt 60 ]; then
    sbatch submit_simulation_6H.slurm ${iteration} ${dense} ${eta}
fi





