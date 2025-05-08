#!/bin/bash -l

# for name in WT dPSQ; do
for name in intra_psq_inter_wt intra_wt_inter_psq; do

home='/home/sb95/ChromatinDynamics/scripts/Drosophila/run_sims'
# opt_lambda=/work/cms16/sb95/Drosophila/Adam/${name}_optimize_eta0.01_init_opt_lambda/input/lambda_10
# opt_lambda=/work/cms16/sb95/Drosophila/${name}_optimize_eta1.0_init_opt_lambda/input/lambda_30
opt_lambda=/work/cms16/sb95/Drosophila/input_data/${name}


path=/work/cms16/sb95/Drosophila/Optimized_runs/${name}/Adam_eta0.01_init_opt_lambda10
rm -r ${path}
mkdir -p -v ${path}

cp -r /home/sb95/ChromatinDynamics/src ${path}
cp run_optimized_simulation.py submit_optimized_runs.slurm ${path}

echo $PWD
echo "#!/bin/bash -l

sbatch submit_optimized_runs.slurm ${opt_lambda}">${path}/sub_slurm.sh
cd ${path}
sh sub_slurm.sh

cd ${home}

done