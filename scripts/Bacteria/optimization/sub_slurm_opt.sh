#!/bin/bash -l
partn="commons"
acnt="commons"
code_home="/home/sb95/ChromatinDynamics/scripts/Bacteria"
opt_file="run_optimization_exponential.py"

for bug in Bsub Ecoli; do

if [[ "$bug" == "Ecoli" ]]; then
rconf=3.0
elif [[ "$bug" == "Bsub" ]]; then
rconf=2.5
fi

for flag in WT matP mukBmatP dparAB mukB dsmc ; do
# Skip dsmc and dparAB for Ecoli

if [[ "$bug" == "Ecoli" && ( "$flag" == "dsmc" || "$flag" == "dparAB" ) ]]; then
    continue
fi

# Skip mukB, matP, and mukBmatP for Bsub
if [[ "$bug" == "Bsub" && ( "$flag" == "mukB" || "$flag" == "matP" || "$flag" == "mukBmatP" ) ]]; then
    continue
fi

echo "Running for $bug $flag"

for zconf in 50.0; do
for eta in 0.5; do
for interOri in 0.0; do

data_home="/scratch/sb95/Bacteria_full_exponential_optimization_pow2/sgd/"$bug"/"$flag"/Optimize/Rconf"$rconf"_zconf"$zconf"_eta"$eta"_interOri${interOri}_expScheduler0.1"
hic_file="/home/sb95/Bacteria_chromosome/hic_expt/HiC_${bug}_${flag}_ori_centered.txt"
rm -r $data_home
mkdir -p -v $data_home
cd $data_home
cp -r /home/sb95/ChromatinDynamics/src $data_home
cp $code_home/$opt_file $data_home

slurm_run="#!/bin/bash -l

#SBATCH --job-name=$bug$flag
#SBATCH --partition=$partn
#SBATCH --account=$acnt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=23:59:00
#SBATCH --export=ALL

module purge
module load GCC/12.3.0  OpenMPI/4.1.5 OpenMM/8.0.0-CUDA-12.1.1 h5py/3.9.0
module load Mamba/23.11.0-0 
source /opt/apps/software/Mamba/23.11.0-0/bin/activate
conda activate /home/sb95/.conda_venv/opmm

python3 $opt_file -exp $hic_file -nsteps 80 -eta $eta -rconf $rconf -zconf $zconf -interOri $interOri

"
runname="sub_opt.slurm"
echo "$slurm_run">$runname
sbatch $runname

done
done
done
done
done