#!/bin/bash

#$ -N kedro
#$ -cwd ../
#$ -V
#$ -o logs/output.o 
#$ -e logs/error.e

kedro run --env mpi_regimeB --pipeline tr_without_pl+mv