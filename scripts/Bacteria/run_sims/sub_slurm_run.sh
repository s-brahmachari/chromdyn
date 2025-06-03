#!/bin/bash -l
partn="commons"
acnt="commons"
code_home="/home/sb95/ChromatinDynamics/scripts/Bacteria/run_sims"
for bug in Ecoli Bsub; do

if [[ "$bug" == "Ecoli" ]]; then
rconf=3.0
rconf_opt=3.0
oriC=0 #313
elif [[ "$bug" == "Bsub" ]]; then
rconf=2.5
rconf_opt=2.5
oriC=0 #202
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
init_ff="/scratch/sb95/Bacteria_full_exponential_optimization_pow2/sgd/"$bug"/"$flag"/Optimize/Rconf"$rconf_opt"_zconf50.0_eta0.5_interOri0.0_expScheduler0.1/input/lambda_25_0.0"

# Skip mukB, matP, and mukBmatP for Bsub
if [[ "$flag" != "mukBmatP" ]]; then
    continue
fi
for zconf in 50.0; do
for repfrac in 0.0; do #0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8; do
for interOri in 0.0; do #-0.45 -0.5; do

if [[ "$repfrac" == "0.0" && "$interOri" != "0.0" ]] || [[ "$repfrac" != "0.0" && "$interOri" == "0.0" ]]; then
        continue
fi

data_home="/scratch/sb95/Bacteria_full_exponential_optimization_pow2/sgd/"$bug"/"$flag"/Optimized_runs_lambda25/Rconf"$rconf"_zconf"$zconf"/Rep_frac${repfrac}/interOri${interOri}"

# rm -r $data_home
mkdir -p -v $data_home
cd $data_home
cp -r /home/sb95/ChromatinDynamics/src $data_home
cp $code_home/run_sim_expon_random_interOri.py $data_home
cp /home/sb95/Bacteria_chromosome/scripts/bacteria_traj_analysis.py $data_home

slurm_run="#!/bin/bash -l

#SBATCH --job-name=$bug-$flag
#SBATCH --partition=$partn
#SBATCH --account=$acnt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=23:59:00
#SBATCH --export=ALL

module purge
module load GCC/12.3.0  OpenMPI/4.1.5 OpenMM/8.0.0-CUDA-12.1.1 h5py/3.9.0
module load Mamba/23.11.0-0 
source /opt/apps/software/Mamba/23.11.0-0/bin/activate
conda activate /home/sb95/.conda_venv/opmm

python3 run_sim_expon_random_interOri.py -nrep_init 20 -rconf $rconf -zconf $zconf -nrep 30 -interOri $interOri -repFrac $repfrac -init_ff $init_ff 
python3 bacteria_traj_analysis.py -rep_frac $repfrac

"
runname="sub_opt.slurm"
echo "$slurm_run">$runname
sbatch $runname

done
done
done
done
done