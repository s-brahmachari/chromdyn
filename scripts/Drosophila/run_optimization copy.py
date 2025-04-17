from Optimization import AdamTraining 

import numpy as np
import pandas as pd
import h5py
import sys
import os
# from scipy import sparse

# Parameters from the submission scritps
iteration   = sys.argv[1]
inputFolder = sys.argv[2]
dense       = sys.argv[3]
eta         = float(sys.argv[4])
replicas    = sys.argv[5:]
# lambdaFile_types  = "input/lambda_types"

# Parameters from the submission scritps
# iteraction = 0
# inputFolder = "lambdas"
# seq = "input/seq_chr10_100k.txt"
# dense = "input/chr10_100k.dense"
# replicas = "1"

lambdaFile = inputFolder + "/lambda_" + str(iteration)


# Parameters for the crosslinking function
#cons_mu = 2.0#3.867 #3.22 
#cons_rc = 2.0#1.681 #1.78


# Parameter for lambda equation
#damp = 3e-8 #3*10**-7

b = AdamTraining(mu =3.22, rc = 1.78, eta=eta, beta1=0.9, beta2=0.999, epsilon=1e-8, it=int(iteration), updateNeeded=True, update_storagePath='update', method='classic')
b.getPars(HiC=dense, norm=False)

# #opt2 = CustomMiChroMTraining(ChromSeq=seq, TypesTable=lambdaFile_types, 
#                              mu=cons_mu, rc=cons_rc, cutoff=0.01,
#                             #  IClist=lambda_old_file, 
#                             #  dinit=dinit, dend=dend
#                              )

for replica in replicas:

    print("Reading replica ", replica)

    with h5py.File(replica + "/Pi_" + str(iteration) + "_" + str(replica.split('_')[-1])+".h5", 'r') as hf:
        b.Pi +=hf["Pi"][()]

    b.NFrames += int(np.loadtxt(replica + "/Nframes_" + str(iteration) + "_" + str(replica.split('_')[-1])))   
  


lamb_new = b.getLamb(Lambdas=inputFolder + "/lambda_" +str(iteration))   

lamb_new.to_csv(inputFolder + "/lambda_" +str(int(iteration)+1), index=False)

with open("error",'a') as tf:
    tf.write("iteration: %f    Error: %f  \n" % (int(iteration), b.error))

# Create directories for saving

if not os.path.exists("hic"):
    os.mkdir("hic")

np.savetxt("hic/probdist_"+ str(iteration), b.Pi/b.NFrames)

