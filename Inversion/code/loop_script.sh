#!/bin/bash

for param in RGI60-05.16433 RGI60-05.00613 RGI60-05.03529 RGI60-05.15375 RGI60-05.00746
do
	echo "Processing $param"
	python3 ../../../igm2.2.1/igm/igm_run.py --lncd_input_file ../Input_data/$param_input.nc --wncd_output_file ../Output_data/"$param"_output_A-44.09749181063853_c-0.18042111520099424_theta-0.17549456809668829_velMult-0.7243338414153321.nc --wts_output_file ../Output_data/"$param"_ts_A-44.09749181063853_c-0.18042111520099424_theta-0.17549456809668829_velMult-0.7243338414153321.nc --iflo_init_arrhenius 44.09749181063853 --iflo_init_slidingco 0.18042111520099424 --theta 0.17549456809668829 --vel_mult 0.7243338414153321 
done
