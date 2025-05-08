import os
from pathlib import Path
SRC=os.path.join(Path.cwd(), 'src')
import sys
sys.path.append(SRC)
import pandas as pd
from Topology import TopologyGenerator
from ChromatinDynamics import ChromatinDynamics
from HiCManager import HiCManager
from Reporters import save_pdb
import numpy as np
import time
import os

# Parameters from the submission scritps
rep              = sys.argv[1]
iteration        = sys.argv[2]
eta              = float(sys.argv[3])

lambdas          = f"input/lambda_{iteration}"
folder           = f"output_{iteration}/run_{rep}/"

start_time = time.time()
mu=2.0
rc=2.0
p=4
print("Generating Topology ...")
# chain_lens = [471, 977, 1205, 27]
chain_lens = [2355, 4881, 6020, 135]
chain_names=['chrX','chr2','chr3','chr4']
# assert sum(chain_lens)==num_beads, 'hic file does not have same dimension as chain_lens'
generator = TopologyGenerator()
generator.gen_top(chain_lens=chain_lens, chain_names=chain_names, types="unique")
# generator.save_top(folder+'topology.txt')

print("Initializing simulation object ...")
sim = ChromatinDynamics(
        topology=generator.topology,
        platform_name="OpenCL", 
        name='droso',
        output_dir=folder,
        console_stream=False,
        )

print("Reading input lambdas ...")
lambda_df = pd.read_csv(lambdas)
interaction_matrix = lambda_df.values
type_labels = list(lambda_df.columns.values)
# print(type_labels, interaction_matrix.shape)
# assert num_beads == interaction_matrix.shape[0] and len(type_labels) == num_beads
# print('setting bond exclusions')
# sim.force_field_manager.exclude_bonds_from_NonBonded=False
print("Adding forces ...")
sim.force_field_manager.add_harmonic_bonds(k=30.0, r0=1.0, group=0)
# sim.force_field_manager.add_harmonic_angles(k=2.0, theta0=180.0, group=1)
sim.force_field_manager.add_self_avoidance(Ecut=4.0, k=5.0, r=0.7, group=2)
sim.force_field_manager.add_type_to_type_interaction(interaction_matrix, type_labels, 
                                                     mu=mu, rc=rc, group=3)
sim.force_field_manager.add_flat_bottom_harmonic(k=0.1, r0=35.0, group=4)

print("Setting up simulation ...")
sim.simulation_setup(
    init_struct='saw3d',
    integrator='langevin',
    temperature=120.0,
    timestep=0.01,
    save_pos=True,
    save_energy=True,
    energy_report_interval=3_000,
    pos_report_interval=1_000,              
    )

print("Running collapse ...")
#collapse
sim.run(1_000_000, report=False)

print("Running sim ...")
for _ in range(10):
    sim.run(1_000_000)
    save_pdb(sim)
    print("saved pdb ...")

sim.save_reports()
print("Simulations done!")

sim_time = time.time() 
elapsed_hours = int((sim_time - start_time) // 3600)
elapsed_minutes = int(((sim_time - start_time)% 3600) // 60)
print(f"Time after simulation {elapsed_hours} hours and {elapsed_minutes} minutes")

print("Generating HiC from traj ...")
hicman = HiCManager(logger=sim.logger)
hic = hicman.gen_hic_from_cndb(traj_file=sim.pos_report_file, mu=mu, rc=rc, p=p, parallel=False)
np.savetxt(os.path.join(sim.output_dir, f"Pi_{str(iteration)}_{str(rep)}.txt"), hic)

print(f"HiC running time: {int(time.time()-sim_time) // 3600} hours {int((time.time()-sim_time)% 3600) // 60} mins")
