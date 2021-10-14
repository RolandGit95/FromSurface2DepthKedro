# ! python

# name
#$ -N kedro

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# serial queue
#$ -q taranis-gpu1.q

# Path for output
#$ -o /logs

import os

os.system("kedro run --env mpi_kedro run --env mpi_regimeB --pipeline tr_without_pl+mv --params data_science.name:STLSTM_t_1_6_12_17_22_27_32_d_1,data_science.depths:[1]")
