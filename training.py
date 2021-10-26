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


import sys
import os
import numpy as np
import kedro


def main(argv):
    print(os.environ['SGE_TASK_ID'])
    SGE_TASK_ID = int(os.environ['SGE_TASK_ID']) - 1

    name = f'STLSTM_t_0_2_4_6_8_10_12_14_16_18_20_22_24_26_28_30_31_d_{SGE_TASK_ID}'
    depths = str([SGE_TASK_ID])
    time_steps = str([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31])
    time_steps = time_steps.replace(',','_')

    ds = 'data_science'
    os.system(f"kedro run --env mpi_regimeB --pipeline tr_without_pl+mv --params {ds}.name:{name},{ds}.depths:{depths},{ds}.time_steps:{time_steps}")


if __name__ == '__main__':
    main(sys.argv[1:])
