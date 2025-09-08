#!/bin/bash

for param in RGI60-05.16433 RGI60-05.00613 RGI60-05.03529 RGI60-05.15375 RGI60-05.00746
do
	echo "Processing $param"
	python3 igm_run --lncd_input_file ../Input_data/"$param"_input.nc --wncd_output_file ../Output_data/"$param"_output_A-40.0_c-0.08_theta-0.1_velMult-1.0.nc --wts_output_file ../Output_data/"$param"_ts_A-40.0_c-0.08_theta-0.1_velMult-1.0.nc --iflo_init_arrhenius 40.0 --iflo_init_slidingco 0.08 --theta 0.1 --vel_mult 1.0 
done
