import os
from pathlib import Path
SRC=os.path.join(Path.cwd(), 'src')
import sys
sys.path.append(SRC)

from Topology import TopologyGenerator
from ChromatinDynamics import ChromatinDynamics
import numpy as np
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('-N',default='200',dest='N_poly', type=int)
parser.add_argument('-chi',default=0.0,dest='chi', type=float)
parser.add_argument('-temp',default=120.0,dest='temp', type=float)
parser.add_argument('-Nrep',default=1,dest='Nrep', type=int)
parser.add_argument('-output',default='output',dest='output', type=str)

args=parser.parse_args()

N_poly = int(args.N_poly)
chi = float(args.chi)
temp = float(args.temp)
Nrep = int(args.Nrep)

generator = TopologyGenerator()
# By default all monomers have type "A"
generator.gen_top([N_poly])
num_blocks = 1000

type_labels = ["A", "B"]
interaction_matrix = [[chi, 0.0], [0.0, 0.0]]

for replica in range(Nrep):
    print(f"Running replica {replica} ... ")
    sim = ChromatinDynamics(
        topology=generator.topology,
        platform_name="OpenCL", 
        name=f'polymer_{replica}',
        output_dir=f"{args.output}",
        console_stream=False,
        )

    sim.force_field_manager.add_harmonic_bonds(k=30.0, r0=1.0, group=0)
    sim.force_field_manager.add_self_avoidance(Ecut=4.0, k=5.0, r=1.0, group=1)
    sim.force_field_manager.add_type_to_type_interaction(interaction_matrix, type_labels, mu=2.0, rc=2.0, group=2)
    
    sim.simulation_setup(
        init_struct='randomwalk',
        integrator='langevin',
        temperature=temp,
        save_pos=False,
        save_energy=True,
        energy_report_interval=1000,                
        )
    
    sim.run(100_000) #relax
    
    for _ in range(num_blocks):
        sim.run(1000)