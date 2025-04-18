#!/bin/bash -l

for name in WT dPSQ; do
iteration='1'
dense=/work/cms16/sb95/Drosophila/input_data/${name}.txt
home='/home/sb95/ChromatinDynamics/scripts/Drosophila/'
init_lambda='/work/cms16/sb95/Drosophila/input_data/lambda_-0.05'

for eta in 1.5 2.0; 
do
path=/work/cms16/sb95/Drosophila/${name}_optimize_eta${eta}_init_-0.05_lambda
rm -r ${path}
mkdir -p -v ${path}/input

cp -r /home/sb95/ChromatinDynamics/src ${path}
cp run_optimization.py run_simulation.py submit_simulation_6H.slurm ${path}
cp ${init_lambda} ${path}/input/lambda_1

echo $PWD
echo "#!/bin/bash -l

sbatch submit_simulation_6H.slurm ${iteration} ${dense} ${eta}">${path}/sub_slurm.sh
cd ${path}
sh sub_slurm.sh

cd ${home}

done
done