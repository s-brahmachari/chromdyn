#!/bin/bash -l

for name in WT dPSQ; do
iteration='1'
dense=/work/cms16/sb95/Drosophila/input_data/${name}_10k.txt
home='/home/sb95/ChromatinDynamics/scripts/Drosophila/optimization_high_res'
# init_lambda=/work/cms16/sb95/Drosophila/input_data/lambda_10k_-0.05 #lambda_${name}_opt_10k #
init_lambda=/work/cms16/sb95/Drosophila/10k/SGD/${name}_optimize_eta0.8_init_opt_lambda/input/lambda_35

for eta in 0.01 0.1; 
do
path=/work/cms16/sb95/Drosophila/10k/Adam/${name}_optimize_eta${eta}_init_opt_lambda35
rm -r ${path}
mkdir -p -v ${path}/input

cp -r /home/sb95/ChromatinDynamics/src ${path}
cp run_optimization.py run_simulation.py submit_simulation.slurm submit_optimization.slurm ${path}
cp ${init_lambda} ${path}/input/lambda_1

echo $PWD
echo "#!/bin/bash -l

sbatch submit_simulation.slurm ${iteration} ${dense} ${eta}">${path}/sub_slurm.sh

cd ${path}
sh sub_slurm.sh

cd ${home}

done
done