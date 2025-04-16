#!/bin/bash -l

counter=0
counter2=0
out_str=""
code_home=/home/sb95/ChromatinDynamics
nrep=12
data_home=/work/cms16/sb95/SAW_globule_extension_const_force_fixed_temp
# rm -r $data_home
mkdir -p -v $data_home
cp -r $code_home/src $data_home
cp $code_home/scripts/run_force_ext_const_force.py $data_home
cd $data_home
rm launcher_jobs_sim*
rm submit_sim_*
for chi in 1.2 1.5; do
for temp in 300; do
for N in 500; do
for fz in $(seq 0.5 0.5 8.2); do

save_folder=$data_home/epsilon_$chi/temp_$temp/N_$N/fz$fz
# mkdir -p -v $save_folder

# if (( $(echo "$fz == 2.5" | bc -l) )); then
#     continue  # Skip this iteration
# fi

if [[ -z "$out_str" ]]; then
    out_str=$"python run_force_ext_const_force.py -N ${N} -chi ${chi} -temp ${temp} -Nrep ${nrep} -output ${save_folder} -fz ${fz}"
else
    out_str+=$'\n'"python run_force_ext_const_force.py -N ${N} -chi ${chi} -temp ${temp} -Nrep ${nrep} -output ${save_folder} -fz ${fz}"
fi

((counter++))
((counter2++))

if (( counter == 8 )); then
# echo "$out_str"
sbatch_file="#!/bin/bash -l

#SBATCH --job-name=fext
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1            # this can be more, up to 22 on aries
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --export=ALL

module purge
module load foss/2020b Launcher_GPU OpenMPI 
source \$HOME/miniforge3/bin/activate
conda activate openmm-env

# Controlling Launcher and showing some job info
export LAUNCHER_WORKDIR=\`pwd\`
export LAUNCHER_JOB_FILE=\$PWD/launcher_jobs_sim$counter2
export LAUNCHER_BIND=1

# Each iteration is an inversion
# rm \${LAUNCHER_WORKDIR}/launcher_jobs_sim &> /dev/null
date
echo \"${out_str}\" >> \${LAUNCHER_WORKDIR}/launcher_jobs_sim$counter2

\$LAUNCHER_DIR/paramrun

wait
date"
echo "$sbatch_file">>submit_sim_$counter2.slurm
sbatch submit_sim_$counter2.slurm
echo "--------------------------------------"
counter=0
out_str=""
fi
done
done
done
done
# Print remaining commands if any
if [[ -n "$out_str" ]]; then
echo "$out_str"
sbatch_file="#!/bin/bash -l

#SBATCH --job-name=fext
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1            # this can be more, up to 22 on aries
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --export=ALL

module purge
module load foss/2020b Launcher_GPU OpenMPI 
source \$HOME/miniforge3/bin/activate
conda activate openmm-env

# Controlling Launcher and showing some job info
export LAUNCHER_WORKDIR=\`pwd\`
export LAUNCHER_JOB_FILE=\$PWD/launcher_jobs_sim
export LAUNCHER_BIND=1

# Each iteration is an inversion
rm \${LAUNCHER_WORKDIR}/launcher_jobs_sim &> /dev/null
date
echo \"${out_str}\" >> \${LAUNCHER_WORKDIR}/launcher_jobs_sim
\$LAUNCHER_DIR/paramrun
wait
date"
echo "$sbatch_file">>submit_sim_$counter2.slurm
sbatch submit_sim_$counter2.slurm
fi
