#!/bin/bash

declare -a names=(STLSTM_t_1_6_12_17_22_27_32_d0
                  STLSTM_t_1_6_12_17_22_27_32_d1
                  STLSTM_t_1_6_12_17_22_27_32_d2
                  STLSTM_t_1_6_12_17_22_27_32_d3)

depths=(0,1,2,3)

#name=$names[$SGE_TASK_ID]
#depth = $depths[$SGE_TASK_ID]

#kedro run --env mpi_regimeB --pipeline tr_without_pl+mv --params prediction.name:$name,validation.name:$name,training.name:$name 