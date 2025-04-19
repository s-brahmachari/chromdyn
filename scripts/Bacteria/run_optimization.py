import os
from pathlib import Path
SRC=os.path.join(Path.cwd(), 'src')
import sys
sys.path.append(SRC)
import numpy as np
import pandas as pd
from Optimization import EnergyLandscapeOptimizer
import argparse as arg
from Topology import TopologyGenerator
from ChromatinDynamics import ChromatinDynamics
from HiCManager import HiCManager
from Utilities import save_pdb
import time

parser=arg.ArgumentParser()
parser.add_argument('-exp',required=True,dest='exp', type=str)
parser.add_argument('-nsteps', default=20,dest='nsteps', type=int)
parser.add_argument('-eta',default=0.02,dest='eta', type=float)
parser.add_argument('-init_ff',default=None,dest='init_ff', type=str)
parser.add_argument('-rconf',default=5.0,dest='rconf', type=float)
parser.add_argument('-zconf',default=10.0,dest='zconf', type=float)
args=parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

eta = args.eta
hic_file = args.exp
iter_steps = args.nsteps
rconf = args.rconf
zconf = args.zconf

mu=2.0
rc=1.5
p=2.0
N_replica=3
assert Path(hic_file).exists(), 'HiC file does not exist!'

#Initialize optimization
opt = EnergyLandscapeOptimizer(eta=eta, it=1, method='sgd', scheduler='exponential', scheduler_decay=0.1)
opt.load_HiC(hic_file=hic_file, neighbors=0)
num_beads = opt.phi_exp.shape[0]

# generate initial lambda

if Path(str(args.init_ff)).exists():
    lambda_df = pd.read_csv(args.init_ff)
    interaction_matrix = lambda_df.values
    type_labels = list(lambda_df.columns.values)
    print("Initial lambda from file:", args.init_ff)
else:
    interaction_matrix = -0.05 * np.ones(shape=(num_beads, num_beads))
    type_labels = [f"M{idx+1}" for idx in range(num_beads)]
    lambda_df = pd.DataFrame(interaction_matrix, columns=type_labels)
    print("Initial lambda -0.05")

input_dir = Path('input/')
input_dir.mkdir(exist_ok=True)
lambda_df.to_csv('input/lambda_1', index=False)

# Generate topology
print("Generating Topology ...")
chain_lens = [num_beads]
chain_names=['chr1']
# assert sum(chain_lens)==num_beads, 'hic file does not have same dimension as chain_lens'
generator = TopologyGenerator()
generator.gen_top(chain_lens=chain_lens, chain_names=chain_names, types=type_labels, isRing=[1])
generator.save_top('input/topology.txt')

for iteration in range(1, iter_steps+1):
    print('-'*30+f"\nStarting iteration {iteration} ...")
    start_time = time.time()
    print("Reading input lambdas ...")
    lambda_t_df = pd.read_csv(f'input/lambda_{iteration}')
    interaction_matrix = lambda_t_df.values
    type_labels = list(lambda_t_df.columns.values)
    
    for replica in range(N_replica):
        print(f"Simulating replica {replica+1} ...")
        print("Initializing simulation object ...")
        sim = ChromatinDynamics(
            topology=generator.topology,
            platform_name="CUDA", 
            name='bacteria',
            output_dir=f'output_{iteration}/run_{replica}',
            console_stream=False,
            )
        
        print('setting bond exclusions')
        sim.force_field_manager.exclude_bonds_from_NonBonded=False
        
        print("Adding forces ...")
        sim.force_field_manager.add_harmonic_bonds(k=30.0, r0=1.0, group=0)
        sim.force_field_manager.add_self_avoidance(Ecut=4.0, k=5.0, r=0.7, group=1)
        sim.force_field_manager.add_type_to_type_interaction(interaction_matrix, type_labels, 
                                                            mu=mu, rc=rc, group=2)
        sim.force_field_manager.add_cylindrical_confinement(r_cyl=rconf, z_cyl=zconf, group=3)

        print("Setting up simulation ...")
        sim.simulation_setup(
            init_struct='saw3d',
            integrator='langevin',
            temperature=120.0,
            timestep=0.01,
            save_pos=True,
            save_energy=True,
            energy_report_interval=3_000,
            pos_report_interval=500,              
            )

        print("Running collapse ...")
        #collapse
        sim.run(500_000, report=False)

        print("Running sim ...")
        for _ in range(5):
            sim.run(1_000_000)
            save_pdb(sim)
            print("saved pdb ...")

        sim.save_reports()
        print("Simulations done!")

        print("Generating HiC from traj ...")
        hicman = HiCManager(logger=sim.logger)
        hic = hicman.gen_hic_from_cndb(traj_file=sim.pos_report_file, mu=mu, rc=rc, p=p, parallel=True)
        np.savetxt(os.path.join(sim.output_dir, f"Pi_{str(iteration)}_{str(replica)}.txt"), hic)
        print('-'*10)
    
    # generate hic from replicas
    hic_sim=None
    idx=0
    for replica in range(N_replica):
        print("Reading replica ", replica)
        hic_file = Path(f'output_{iteration}') / f'run_{replica}' / f"Pi_{str(iteration)}_{replica}.txt"
        if not hic_file.exists(): continue
        try:
            if hic_sim is None:    
                hic_sim = np.loadtxt(hic_file)
                idx+=1
            else:
                hic_sim += np.loadtxt(hic_file)
                idx+=1
        except Exception as e:
                print(f"Something went wrong with numpy loadtxt: {e}")
    assert idx>0, "no HiC could be loaded"
    hic_sim /= idx
    
    Path("hic_sim").mkdir(exist_ok=True)
    np.savetxt("hic_sim/probdist_" + str(iteration) + ".txt", hic_sim)
    
    lambda_t_file = Path('input') / f"lambda_{str(iteration)}"
    assert lambda_t_file.exists(), f'lambda input file is missing for iteration {iteration}'
    lambda_t_df = pd.read_csv(lambda_t_file, sep=None, engine='python')
    assert lambda_t_df.values.shape[0]==hic_sim.shape[0], f'lambda and hicsim shape mismatch'
    updated_lambda = opt.get_updated_params(lambda_t_df.values, hic_sim)
    updated_lambda_df = pd.DataFrame(updated_lambda, columns=list(lambda_t_df.columns.values))
    updated_lambda_df.to_csv('input' + "/lambda_" +str(int(iteration)+1), index=False)

    with open("error",'a') as err:
        err.write("iteration: %f    Error: %f  \n" % (int(iteration), opt.error))

    sim_time = time.time() 
    elapsed_hours = int((sim_time - start_time) // 3600)
    elapsed_minutes = int(((sim_time - start_time)% 3600) // 60)
    print(f"Time for iteration {iteration}: {elapsed_hours} hours and {elapsed_minutes} minutes")

