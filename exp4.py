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
    str_time_steps = "["

    for time_step in time_steps:
        name += str(time_step) + "_"
        str_time_steps += str(time_step) + "_"

    str_time_steps = str_time_steps[:-1] + "]"
    name += f"d_{depth}"

    return time_steps, name, str_time_steps

#time_steps, name, str_time_step = getTimeStepsName(2, 31)
#print(time_steps, name, str_time_step)

# %%
def main():
    print(os.environ['SGE_TASK_ID'])
    SGE_TASK_ID = int(os.environ['SGE_TASK_ID'])

    dt = 2
    depth = SGE_TASK_ID

    time_steps, name, str_time_step  = getTimeStepsName(dt, depth)

    depths = f"[{depth}]"
    os.system(f"kedro run --env exp4_mpi --pipeline tr_without_pl+mv --params data_science.name:{name},data_science.depths:{depths},data_science.time_steps:{str_time_step}")

if __name__=='__main__':
    main()


