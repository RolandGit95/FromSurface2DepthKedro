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

def main():
    print(os.environ['SGE_TASK_ID'])
    SGE_TASK_ID = int(os.environ['SGE_TASK_ID'])

    name = f'STLSTM_t32_d_{SGE_TASK_ID}'
    depths = f"[{SGE_TASK_ID}]"
    os.system(f"kedro run --env exp4_mpi --pipeline tr_without_pl+mv --params data_science.name:{name},data_science.depths:{depths}")

if __name__=='__main__':
    main()
