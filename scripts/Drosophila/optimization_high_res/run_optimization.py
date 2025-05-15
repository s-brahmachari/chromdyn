import os
from pathlib import Path
SRC=os.path.join(Path.cwd(), 'src')
import sys
sys.path.append(SRC)
import numpy as np
import pandas as pd
from Optimization import EnergyLandscapeOptimizer
from HiCManager import HiCManager

# Parameters from the submission scritps
iteration   = sys.argv[1]
inputFolder = sys.argv[2]
hic_exp_file = sys.argv[3]
eta         = float(sys.argv[4])
replica_folders    = sys.argv[5:]

if __name__ == "__main__":
    mu=2.0
    rc=2.0
    p=4.0

    print("Generating HiC from traj ...")
    hicman = HiCManager(logger=None)

    # generate hic from replicas
    hic_sim=None
    idx=0
    for replica in replica_folders:
        print("Reading replica ", replica)
        traj_file = Path(replica) / f"droso_positions.cndb"
        try:
            if hic_sim is None:    
                hic_sim = hicman.gen_hic_from_cndb(traj_file=str(traj_file), mu=mu, rc=rc, p=p, parallel=True, skip_frames=10)
                idx+=1
            else:
                hic_sim += hicman.gen_hic_from_cndb(traj_file=str(traj_file), mu=mu, rc=rc, p=p, parallel=True, skip_frames=10)
                idx+=1
        except Exception as e:
                print(f"Something went wrong with numpy loadtxt: {e}")
    assert idx>0, "no HiC could be loaded"
    hic_sim /= idx

    np.savetxt("hic_sim/probdist_"+ str(iteration), hic_sim)
    
    opt = EnergyLandscapeOptimizer(eta=eta, it=iteration, method='sgd', scheduler='none', scheduler_decay=0.01)
    opt.load_HiC(hic_file=hic_exp_file, neighbors=0, filter='median')
    
    if int(iteration)>1:
        param_file = f'opt_params/sgd_params_{int(iteration)-1}.h5'
        assert Path(param_file).exists(), 'Param file does not exist'
        opt.set_optimization_params(param_file)
    else:
        opt.init_optimization_params()
        
    num_beads = opt.phi_exp.shape[0]

    lambda_t_file = Path(inputFolder) / f"lambda_{str(iteration)}"
    assert lambda_t_file.exists(), f'lambda input file is missing for iteration {iteration}'
    lambda_t_df = pd.read_csv(lambda_t_file, sep=None, engine='python')
    assert lambda_t_df.values.shape[0]==hic_sim.shape[0], f'lambda and hicsim shape mismatch'
    updated_lambda_sgd = opt.get_updated_params(lambda_t_df.values, hic_sim)
    
    opt.set_opt_method("adam")
    opt.eta0 = 0.1 * eta
    opt.t -= 1
    if int(iteration)>1:
        param_file = f'opt_params/adam_params_{int(iteration)-1}.h5'
        assert Path(param_file).exists(), 'Param file does not exist'
        opt.set_optimization_params(param_file)
    else:
        opt.init_optimization_params()
        
    updated_lambda_adam = opt.get_updated_params(lambda_t_df.values, hic_sim)
    
    updated_lambda = updated_lambda_adam + updated_lambda_sgd
    
    updated_lambda_df = pd.DataFrame(updated_lambda, columns=list(lambda_t_df.columns.values))
    updated_lambda_df.to_csv(inputFolder + "/lambda_" +str(int(iteration)+1), index=False)
    error = opt.get_error(hic_sim)
    
    with open("error",'a') as tf:
        tf.write("iteration: %f    Error: %f  \n" % (int(iteration), error))

    if not os.path.exists("hic_sim"):
        os.mkdir("hic_sim")


