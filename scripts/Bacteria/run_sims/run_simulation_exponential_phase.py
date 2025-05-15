import os
from pathlib import Path
SRC=os.path.join(Path.cwd(), 'src')
import sys
sys.path.append(SRC)
import numpy as np
import pandas as pd
import argparse as arg
from Topology import TopologyGenerator
from ChromatinDynamics import ChromatinDynamics
from HiCManager import HiCManager
from Reporters import save_pdb
import time

parser=arg.ArgumentParser()

parser.add_argument('-init_ff',default=None,dest='init_ff', type=str)
parser.add_argument('-rconf',default=5.0,dest='rconf', type=float)
parser.add_argument('-zconf',default=10.0,dest='zconf', type=float)
parser.add_argument('-repFrac',default=0.0,dest='repFrac', type=float)
parser.add_argument('-interOri',default=0.0,dest='interOri', type=float)
parser.add_argument('-oriC',default=0,dest='oriC', type=int)
parser.add_argument('-nrep',default=1,dest='nrep', type=int)

args=parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

rconf = args.rconf
zconf = args.zconf
rep_frac = args.repFrac
oriC = args.oriC
N_replica=args.nrep
inter_ori = args.interOri

mu=2.0
rc=1.5
p=2.0

start_time = time.time() 

# generate lambda
if Path(str(args.init_ff)).exists():
    lambda_df = pd.read_csv(args.init_ff)
    # interaction_matrix = lambda_df.values
    # type_labels = list(lambda_df.columns.values)
    num_beads = lambda_df.values.shape[0]
    print("Optimized lambda from file:", args.init_ff)
    print("polymer size: ", num_beads)
else:
    print("No initial lambda provided!")
    raise ValueError

rep_len = int(rep_frac * num_beads)
#shift such that origin is centralized
shift = int(num_beads//2 - oriC)
ff_chain = np.roll(lambda_df.values, shift, axis=(0,1))
print("Shifted optimized chain by ", shift)

if rep_len>0:
    start = (num_beads-rep_len)//2
    end = (num_beads+rep_len)//2
    ff_rep = ff_chain[start:end, start:end]
    foriter1 = ff_chain[start:end,:start]
    foriter2 = ff_chain[end:,start:end]
    ff_exp_phase = np.zeros(shape=(num_beads+rep_len, num_beads+rep_len))
    
    ff_exp_phase[:num_beads, :num_beads] = ff_chain
    ff_exp_phase[num_beads:, num_beads:] = ff_rep
    ff_exp_phase[num_beads:, :start] = foriter1
    ff_exp_phase[num_beads:, start:end] = inter_ori * np.ones(shape=(rep_len, rep_len))
    ff_exp_phase[num_beads:, end:num_beads] = np.rot90(foriter2, 1)
    ff_exp_phase = np.triu(ff_exp_phase) + np.triu(ff_exp_phase).T
    chain_lens = [num_beads, rep_len]
    chain_names=['M', 'D']
    isRing=[1, 0]
else:
    ff_exp_phase = np.triu(ff_chain) + np.triu(ff_chain).T
    chain_lens = [num_beads]
    chain_names=['M']
    isRing=[1]


type_labels = [f'M{idx}' for idx in range(1, num_beads+1)] + [f'D{idx}' for idx in range(1, rep_len+1)]
ff_exp_df = pd.DataFrame(ff_exp_phase, columns=type_labels)

input_dir = Path('input/')
input_dir.mkdir(exist_ok=True)
ff_exp_df.to_csv('input/lambda_exp', index=False)

# Generate topology
print("Generating Topology ...")
# assert sum(chain_lens)==num_beads, 'hic file does not have same dimension as chain_lens'
generator = TopologyGenerator()
generator.gen_top(chain_lens=chain_lens, chain_names=chain_names, types=type_labels, isRing=isRing)

if rep_len>0:
    # add mother daughter connections
    di = list(list(generator.topology.chains())[1].atoms())[0]
    df = list(list(generator.topology.chains())[1].atoms())[-1]
    m1 = list(list(generator.topology.chains())[0].atoms())[num_beads//2 - rep_len//2]
    m2 = list(list(generator.topology.chains())[0].atoms())[num_beads//2 + rep_len//2]
    generator.topology.addBond(di,m1)
    generator.topology.addBond(df,m2)

generator.save_top('input/topology.txt')

for replica in range(N_replica):
    print(f"Simulating replica {replica+1} ...")
    print("Initializing simulation object ...")
    sim = ChromatinDynamics(
        topology=generator.topology,
        platform_name="CUDA", 
        name='bacteria',
        output_dir=f'run_{replica}',
        console_stream=False,
        )
    
    # print('setting bond exclusions')
    # sim.force_field_manager.exclude_bonds_from_NonBonded=False
    
    print("Adding forces ...")
    sim.force_field_manager.add_harmonic_bonds(k=30.0, r0=1.0, group=0)
    sim.force_field_manager.add_self_avoidance(Ecut=4.0, k=5.0, r=0.7, group=1)
    sim.force_field_manager.add_type_to_type_interaction(ff_exp_phase, type_labels, 
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
        pos_report_interval=1000,              
        )

    print("Running collapse ...")
    #collapse
    sim.run(500_000, report=False)

    print("Running sim ...")
    for _ in range(10):
        sim.run(1_000_000)
        save_pdb(sim)
        print("saved pdb ...")

    sim.save_reports()
    print("Simulations done!")

    print("Generating HiC from traj ...")
    hicman = HiCManager(logger=sim.logger)
    hic = hicman.gen_hic_from_cndb(traj_file=sim.pos_report_file, mu=mu, rc=rc, p=p, parallel=True)
    np.savetxt(os.path.join(sim.output_dir, f"Pi_{str(replica)}.txt"), hic)
    print('-'*10)

# generate hic from replicas
hic_sim=None
idx=0
for replica in range(N_replica):
    print("Reading replica ", replica)
    hic_file = Path(f'run_{replica}') / f"Pi_{replica}.txt"
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
np.savetxt("hic_sim/probdist.txt", hic_sim)

sim_time = time.time() 
elapsed_hours = int((sim_time - start_time) // 3600)
elapsed_minutes = int(((sim_time - start_time)% 3600) // 60)
print(f"Time for runs: {elapsed_hours} hours and {elapsed_minutes} minutes")

