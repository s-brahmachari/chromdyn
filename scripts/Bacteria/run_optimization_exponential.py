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
from Reporters import save_pdb
import time

parser=arg.ArgumentParser()
parser.add_argument('-exp',required=True,dest='exp', type=str)
parser.add_argument('-nsteps', default=20,dest='nsteps', type=int)
parser.add_argument('-eta',default=0.02,dest='eta', type=float)
parser.add_argument('-init_ff',default=None,dest='init_ff', type=str)
parser.add_argument('-rconf',default=5.0,dest='rconf', type=float)
parser.add_argument('-zconf',default=10.0,dest='zconf', type=float)
parser.add_argument('-interOri',default=0.0,dest='interOri', type=float)
parser.add_argument('-oriC',default=None,dest='oriC')
args=parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

eta = args.eta
hic_file = args.exp
iter_steps = args.nsteps
rconf = args.rconf
zconf = args.zconf
oriC = args.oriC
inter_ori = args.interOri
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
    lambda_0 = pd.read_csv(args.init_ff)
    interaction_matrix = lambda_0.values
    type_labels = list(lambda_0.columns.values)
    print("Initial lambda from file:", args.init_ff)
else:
    interaction_matrix = -0.05 * np.ones(shape=(num_beads, num_beads))
    type_labels = [f"M{idx+1}" for idx in range(num_beads)]
    lambda_0 = pd.DataFrame(interaction_matrix, columns=type_labels)
    print("Initial lambda -0.05")

input_dir = Path('input/')
input_dir.mkdir(exist_ok=True)
lambda_0.to_csv('input/lambda_1_0.0', index=False)

def prepare_chain(rep_frac, lambda_df):
    num_beads = lambda_df.values.shape[0]
    rep_len = int(rep_frac * num_beads)
    rep_len += rep_len % 2 #make the length even for easier handling
    
    #shift such that origin is centralized
    if oriC is None:
        shift=0
    elif type(oriC)==int:
        assert oriC<=num_beads, 'wrong Ori coordinate'
        shift = int(num_beads//2 - oriC)
    ff_chain = np.roll(lambda_df.values, shift, axis=(0,1))
    print("Shifted optimized chain by ", shift)
    print("Generating exponential phase interaction matrix ...")
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
        
        ff_exp_phase = np.tril(ff_exp_phase) + np.tril(ff_exp_phase).T
        chain_lens = [num_beads, rep_len]
        chain_names=['M', 'D']
        isRing=[1, 0]
    else:
        ff_exp_phase = np.triu(ff_chain) + np.triu(ff_chain).T
        chain_lens = [num_beads]
        chain_names=['M']
        isRing=[1]
    type_labels = [f'M{idx}' for idx in range(1, num_beads+1)] + [f'D{idx}' for idx in range(1, rep_len+1)]
    lambda_df_exp = pd.DataFrame(ff_exp_phase, columns=type_labels)
    
    print("Generating Topology ...")
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

    return generator, lambda_df_exp

repfrac_vals=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for iteration in range(1, iter_steps+1):
    print('-'*30+f"\nStarting iteration {iteration} ...")
    start_time = time.time()
    print("Reading input lambda ...")
    lambda_t_df = pd.read_csv(f'input/lambda_{iteration}_0.0')
    
    for repfrac in repfrac_vals:
        # continue
        print(f"Simulating repfrac {repfrac} ...")
        
        generator, lambda_t_df_exp = prepare_chain(repfrac, lambda_t_df)
        interaction_matrix = lambda_t_df_exp.values
        type_labels = list(lambda_t_df_exp.columns.values)
        lambda_t_df_exp.to_csv(f'input/lambda_{iteration}_{repfrac}', index=False)
        
        print("Initializing simulation object ...")
        sim = ChromatinDynamics(
            topology=generator.topology,
            platform_name="CUDA", 
            name='bacteria',
            output_dir=f'output_{iteration}/run_{repfrac}',
            console_stream=False,
            )
        
        # print('setting bond exclusions')
        # sim.force_field_manager.exclude_bonds_from_NonBonded=False
        
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
        sim.run(500_000, report=False)#500_000

        print("Running sim ...")
        for _ in range(3):
            sim.run(100_000)#100_000
            save_pdb(sim)
            print("saved pdb ...")

        sim.save_reports()
        print("Simulations done!")

        print("Generating HiC from traj ...")
        hicman = HiCManager(logger=sim.logger)
        hic = hicman.gen_hic_from_cndb(traj_file=sim.pos_report_file, mu=mu, rc=rc, p=p, parallel=True)
        np.savetxt(os.path.join(sim.output_dir, f"Pi_{str(iteration)}_{str(repfrac)}.txt"), hic)
        print('-'*10)
    
    # generate hic from repfracs
    hic_sim=np.zeros(shape=(num_beads,num_beads))
    idx=0
    for repfrac in repfrac_vals:
        print("Reading repfrac ", repfrac)
        hic_file = Path(f'output_{iteration}') / f'run_{repfrac}' / f"Pi_{str(iteration)}_{repfrac}.txt"
        hic_repfrac = np.loadtxt(hic_file)
        rep_len = hic_repfrac.shape[0] - num_beads
        hic_mother = hic_repfrac[:num_beads,:num_beads]
        if rep_len>1:
            hic_daughter=hic_repfrac[num_beads:,num_beads:]
            start = num_beads//2 - rep_len//2
            end = num_beads//2 + rep_len//2
            hic_daughter_ori_ter_left = hic_repfrac[num_beads:, :start]
            hic_daughter_ori_ter_right = hic_repfrac[num_beads:, end:num_beads]
            print(rep_len, start, end)
            # Average the central block
            hic_mother[start:end, start:end] = 0.5 * (hic_mother[start:end, start:end] + hic_daughter)
            hic_mother[start:end, :start] = 0.5 * (hic_mother[start:end, :start] + hic_daughter_ori_ter_left)
            hic_mother[end:num_beads, start:end] = 0.5 * (hic_mother[end:num_beads, start:end] + np.rot90(hic_daughter_ori_ter_right, 1))
        hic_mother = np.tril(hic_mother) + np.tril(hic_mother).T
        np.fill_diagonal(hic_mother, 1.0)
        hic_sim += hic_mother
        idx+=1
    hic_sim /= idx
    
    Path("hic_sim").mkdir(exist_ok=True)
    np.savetxt("hic_sim/probdist_" + str(iteration) + ".txt", hic_sim)
    
    lambda_t_file = Path('input') / f"lambda_{iteration}_0.0"
    assert lambda_t_file.exists(), f'lambda input file is missing for iteration {iteration}'
    lambda_t_df = pd.read_csv(lambda_t_file, sep=None, engine='python')
    assert lambda_t_df.values.shape[0]==hic_sim.shape[0], f'lambda and hicsim shape mismatch'
    updated_lambda = opt.get_updated_params(lambda_t_df.values, hic_sim)
    updated_lambda_df = pd.DataFrame(updated_lambda, columns=list(lambda_t_df.columns.values))
    updated_lambda_df.to_csv(f'input/lambda_{iteration+1}_0.0', index=False)

    with open("error",'a') as err:
        err.write("iteration: %f    Error: %f  \n" % (int(iteration), opt.error))

    sim_time = time.time() 
    elapsed_hours = int((sim_time - start_time) // 3600)
    elapsed_minutes = int((sim_time - start_time) // 60)
    print(f"Time for iteration {iteration}: {elapsed_hours} hours and {elapsed_minutes - elapsed_hours*60} minutes")
