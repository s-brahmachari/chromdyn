from ChromDynamics import MiChroM 
from Optimization import AdamTraining 

import sys
import numpy as np
import h5py
import time

# Parameters from the submission scritps
rep              = sys.argv[1]
iteration        = sys.argv[2]
denseFile        = sys.argv[3]
eta              = float(sys.argv[4])
lambdas          = f"input/lambda_{iteration}"
sequence         = f'input/seq_drosophila'
folder           = f"output_{iteration}/run_{rep}/"


## include in the submission scripts
sim_name     = "opt"
gpu_platform = "OpenCL"

# Parameters for the crosslinking function
cons_mu = 3.22
cons_rc = 1.78

# Simulation time for Collapse
block_collapse    = 1000
n_blocks_collapse = 200 #200

# Simulation time to feed Optimization
block_opt    = 1500
n_blocks_opt = 4000 #5000

#
b = AdamTraining(mu =3.22, rc = 1.78, eta=eta, beta1=0.9, beta2=0.999, epsilon=1e-8, it=int(iteration), updateNeeded=True, update_storagePath='update', method='classic')
b.getPars(HiC=denseFile, norm=False)


# Setup MiChroM object
#
sim = MiChroM(name=sim_name, temperature=1.0, time_step=0.01)
sim.setup(platform=gpu_platform)
sim.saveFolder(folder)

chr_structure = sim.createRandomWalk(ChromSeq=sequence)

sim.loadStructure(chr_structure, center=True)

# Adding Potentials subsection
break_points = [471, 1448, 2653]

for j in range(2679):
    if j in break_points: continue
    else: sim.addHarmonicBond_ij(j, j + 1, kfb=50.0, r0=1.0)

sim.metadata["HarmonicBond"] = repr({"kfb": 50.0})

# sim.addAngles(ka=2.0, theta_rad=2.0*np.arcsin(0.85))
sim.addSelfAvoidance(Ecut=4.0, k_rep=10.0, r0=1.0)
# sim.addGaussianSelfAvoidance(Ecut=args.Ecut, r0=args.rsa)
# sim.addRepulsiveSoftCore(Ecut=4.0)
sim.addCustomTypes(mu=cons_mu, rc = cons_rc, TypesTable=lambdas)
sim.addFlatBottomHarmonic(kr=0.1, n_rad=12)

block    = block_collapse
n_blocks = n_blocks_collapse

start_time = time.time()
for _ in range(n_blocks):
    sim.runSimBlock(block, increment=True)

#sim.removeFlatBottomHarmonic()
elapsed_time = time.time() - start_time
print(f"Time to equilibrate {elapsed_time/60} minutes")
block    = block_opt
n_blocks = n_blocks_opt
start_time = time.time()
for _ in range(n_blocks):
    # perform 1 block of simulation
    sim.runSimBlock(block, increment=True)
    state = sim.getPositions()
    b.probCalc(state)

elapsed_time = time.time() - start_time
elapsed_hours = int(elapsed_time // 3600)
elapsed_minutes = int((elapsed_time % 3600) // 60)
print(f"time after simulation {elapsed_hours} hours and {elapsed_minutes} minutes")
# save final structure and close traj file

with h5py.File(sim.folder + "/Pi_" + str(iteration) + "_" + str(rep) + ".h5", 'w') as hf:
    hf.create_dataset("Pi",  data=b.Pi)

np.savetxt(sim.folder + "/Nframes_" + str(iteration) + "_" + str(rep), [b.NFrames])
