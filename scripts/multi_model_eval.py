# -*- coding: utf-8 -*-
# %%

# ! python

# name
#$ -N eval

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# Provide path to python executable, has to provide the packages from requirements.txt
#$ -S /home/stenger/smaxxhome/anaconda3/envs/pydiver/bin/python

# serial queue
#$ -q taranis-gpu1.q


import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Neural Networks, the Barkley Diver')

    ### Names ###
    parser.add_argument('-name', '--name', type=str, default="STLSTM_t32_d32")
    parser.add_argument('-depths', '--depths', type=list)