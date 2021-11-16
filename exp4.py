#! python
# %%
# name
#$ -N kedro

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# Provide path to python executable
#$ -S /home/stenger/smaxxhome/anaconda3/envs/pydiver/bin/python

# Merge error and out
#$ -j yes

# serial queue
#$ -q taranis-gpu1.q
# -q teutates.q
# -q grannus.q

# Path for output
#$ -o /data.bmp/heart/DataAnalysis/2020_3DExMedSurfaceToDepth/FromSurface2DepthKedro/logs

# job array of length 1
# -t 1:2
# %%

import os
import numpy as np

# %%

def getTimeStepsName(dt, depth):
    time_steps = np.arange(0,32,dt)
    name = "STLSTM_t_"
    for time_step in time_steps:
        name += str(time_step) + "_"
    name += f"d_{depth}"

    return time_steps, name

#time_steps, name = getTimeStepsName(2, 31)
#print(time_steps, name)

# %%
def main():
    print(os.environ['SGE_TASK_ID'])
    SGE_TASK_ID = int(os.environ['SGE_TASK_ID'])

    dt = 2
    depth = SGE_TASK_ID

    time_steps, name = getTimeStepsName(dt, depth)

    depths = f"[{depth}]"
    os.system(f"kedro run --env exp4_mpi --pipeline tr_without_pl+mv --params data_science.name:{name},data_science.depths:{depths},data_science.time_steps:{time_steps}")

if __name__=='__main__':
    main()


