import os
from pathlib import Path
SRC=os.path.join(Path.cwd(), 'src')
import sys
sys.path.append(SRC)
import numpy as np
import pandas as pd
from Optimization import EnergyLandscapeOptimizer

# Parameters from the submission scritps
iteration   = sys.argv[1]
inputFolder = sys.argv[2]
hic_exp_file = sys.argv[3]
eta         = float(sys.argv[4])
replica_folders    = sys.argv[5:]

# generate hic from replicas
hic_sim=None
idx=0
for replica in replica_folders:
    print("Reading replica ", replica)
    hic_file = Path(replica) / f"Pi_{str(iteration)}_{str(replica.split('_')[-1])}.txt"
    # print(hic_file, hic_sim)
    if not hic_file.exists(): continue
    try:
        if hic_sim:    
            print(hic_sim)
            hic_sim += np.loadtxt(hic_file)
            idx+=1
        else:
            hic_sim = np.loadtxt(hic_file)
            idx+=1
    except Exception as e:
            print(f"Something went wrong with numpy loadtxt: {e}")
assert idx>0, "no HiC could be loaded"
hic_sim /= idx

opt = EnergyLandscapeOptimizer(eta=eta, it=iteration, method='sgd', scheduler='exponential', scheduler_decay=0.1)
opt.load_HiC(hic_file=hic_exp_file, neighbors=0, filter='median')
num_beads = opt.phi_exp.shape[0]

lambda_t_file = Path(inputFolder) / f"lambda_{str(iteration)}"
assert lambda_t_file.exists(), f'lambda input file is missing for iteration {iteration}'
lambda_t_df = pd.read_csv(lambda_t_file, sep=None, engine='python')
assert lambda_t_df.values.shape[0]==hic_sim.shape[0], f'lambda and hicsim shape mismatch'
updated_lambda = opt.get_updated_params(lambda_t_df.values, hic_sim)
updated_lambda_df = pd.DataFrame(updated_lambda, columns=list(lambda_t_df.columns.values))
updated_lambda_df.to_csv(inputFolder + "/lambda_" +str(int(iteration)+1), index=False)

with open("error",'a') as tf:
    tf.write("iteration: %f    Error: %f  \n" % (int(iteration), opt.error))

if not os.path.exists("hic_sim"):
    os.mkdir("hic_sim")

np.savetxt("hic_sim/probdist_"+ str(iteration), hic_sim)

