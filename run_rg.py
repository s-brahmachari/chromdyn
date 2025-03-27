import sys
import os
sys.path.append('./ChromatinDynamics')
from topology import TopologyGenerator
from ChromatinDynamics import ChromatinDynamics
import numpy as np
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('-N',default='200',dest='N_poly', type=int)
parser.add_argument('-mode',default='gauss',dest='mode', type=str)
parser.add_argument('-kbond',default=30.0,dest='kbond', type=float)
parser.add_argument('-rbond',default=1.0,dest='rbond', type=float)
parser.add_argument('-kangle',default=2.0,dest='kangle', type=float)
parser.add_argument('-theta0',default=180.0,dest='theta0', type=float)
parser.add_argument('-krep',default=5.0,dest='krep', type=float)
parser.add_argument('-Erep',default=4.0,dest='Erep', type=float)
parser.add_argument('-rrep',default=1.0,dest='rrep', type=float)
parser.add_argument('-chi',default=0.0,dest='chi', type=float)
parser.add_argument('-Nrep',default=1,dest='Nrep', type=int)
parser.add_argument('-output',default='./',dest='output', type=str)

args=parser.parse_args()

k_bond = float(args.kbond)
N_poly = int(args.N_poly)
mode = str(args.mode)
chi = float(args.chi)
r_bond = float(args.rbond)
k_angle =float(args.kangle)
theta0 = float(args.theta0)
k_rep = float(args.krep)
E_rep = float(args.Erep)
r_rep = float(args.rrep)    
output = str(args.output)
Nrep = int(args.Nrep)

generator = TopologyGenerator()
generator.gen_top([N_poly])
Rg = [] 
num_blocks = 100

for replica in range(Nrep):
    sim = ChromatinDynamics(generator.topology, integrator='langevin', platform_name="OpenCL", output_dir=f"{output}", log_file=f"chrom_dynamics_{replica}.log")
    sim.system_setup(mode=mode, k_bond=k_bond, r_bond=r_bond, chi=chi, k_angle=k_angle, theta0=theta0, E_rep=E_rep, r_rep=r_rep, k_rep=k_rep,)
    sim.simulation_setup()
    sim.run(100_000) #relax
    for _ in range(num_blocks):
        sim.run(2000)
        sim.print_force_info()
        Rg.append(sim.analyzer.compute_RG())
np.savetxt(os.path.join(output,"Radius_of_gyration.txt"), Rg)
