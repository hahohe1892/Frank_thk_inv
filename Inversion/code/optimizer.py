from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import subprocess
import shutil
import os
import pandas as pd
import numpy as np
import glob
import rioxarray as rioxr

### set optimization params ###
RIDs = glob.glob('../Input_data/*_input.nc') # search for input data files in ../Input_data/
RIDs = [x.split('/')[2].split('_input.nc')[0] for x in RIDs]

next_run = {}
next_run['glaciers'] =  RIDs
next_run['experiments'] = {}

n_iter = 5 # number of calibration iterations

old_log_paths = [] # if continuing calibration from previous iterations, give path to previous optimization_logs.log file

params = {
    'given_points': # enter key with value of parameters you definitely want to explore
    {
        'A': [40]
    },
    'default': # defaults for parameters used besides given_points in initial exploration
    {
        'A': 20,
        'c': 0.08,
        'theta': 0.1,
        'velMult': 1.0
    },
    'pbounds': # enter bounds within which to search for parameter values
    {
        'A': (10, 78),
        'theta': (0.05, 0.3),
        'c': (0.01, 0.4),
        'velMult': (.5, 1.5)
    }
}

##############################

def full_optimization(next_run, params, n_iter, old_log_paths = [], kappa = 10):

    '''
    full calibration loop with initial probing of given parameters, followed by 
    exploration of parameters suggested by Bayesian optimizer
    '''
    
    acq = acquisition.UpperConfidenceBound(kappa=kappa)
    optimizer = BayesianOptimization(
        f=None,
        acquisition_function=acq,
        pbounds=params['pbounds'],
        verbose=2,
        random_state=1,
    )

    if len(old_old_paths) > 0:
        for o,old_log_path in enumerate(old_log_paths):
            shutil.copyfile(old_log_path, './old_log{}.log'.format(o))
        load_logs(optimizer, logs=old_log_paths)
        print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

    logger = JSONLogger(path="./optimization_logs.log")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    if len(params['given_points']) > 0:
        initial_probing(optimizer, params)

    update_loop(optimizer, n_iter)
    
def initial_probing(optimizer, params):

    '''
    probe each parameter in given_points together with the default values from the other parameters
    '''
    
    # probe a few given points
    print('Probing given points..')
    for param in params['given_points'].keys():
        for given_point in params['given_points'][param]:

            next_run['experiments'] = params['default']

            # set param from given point
            next_run['experiments'][param] = float(given_point)
            
            # given points should be float
            next_run['experiments'] = {k: float(next_run['experiments'][k]) for k in next_run['experiments']}
            # sort given points alphabetically
            next_run['experiments'] = dict(sorted(next_run['experiments'].items()))

            print(next_run['experiments'])
            
            blackbox_function(next_run)

            target = target_function(next_run)

            print("Found the target value to be:", target)

            optimizer.register(
                params=next_run['experiments'],
                target=target,
            )

def update_loop(optimizer, n_iter):

    '''
    does n_iter iterations with new parameter values suggested by Bayesian optimizer
    '''

    for _ in range(n_iter):
        print('Running iteration {} out of {}'.format(_, n_iter))
              
        next_point_to_probe = optimizer.suggest()
        print("Next point to probe is:", next_point_to_probe)

        next_run['experiments'] =  {k: float(next_point_to_probe[k]) for k in next_point_to_probe.keys()}


        blackbox_function(next_run)


        target = target_function(next_run)

        print("Found the target value to be:", target)

        optimizer.register(
            params=next_point_to_probe,
            target=target,
        )

        print(target, next_point_to_probe)


    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)


def blackbox_function(next_run):

    '''
    Blackbox function to be minimized: here, thickness inversion over a set of glaciers
    '''
    
    create_script(**next_run)
    subprocess.check_call(['chmod', '+x', 'loop_script.sh'])
    subprocess.call(['sh', './loop_script.sh'])

def target_function(next_run):

    '''
    return cost associated with these parameters
    '''
    
    params_string = '_'.join(['{}-{}'.format(x, next_run['experiments'][x]) for x in next_run['experiments']])
    output_files = ['../Output_data/' + x + '_output_{}.nc'.format(params_string) for x in next_run['glaciers']]

    target = calculate_cost(output_files)

    return target

def calculate_cost(output_files, L_MAE_f = 1.0, L_ME_f = 1.5, L_vel_f = 0.5):

    '''
    calculate cost using parameters (eq. (4) in paper). 
    '''
    
    MAEs = []
    MEs = []
    mean_thk_obs = []
    mean_thk_mod = []
    mean_vel_obs = []
    mean_vel_mod = []
    vel_areas = []
    n_obs = []

    for out_file in np.unique(output_files):
        out = rioxr.open_rasterio(out_file)

        # thicknesses
        thk_obs = out.thkobs[0].data
        thk_mod = out.thk[-1].data

        # velocities
        mask = out.mask[-1].data
        slidingco = out.slidingco[-1].data
        vel_obs = out.velsurf_magobs[-1].data
        vel_mod = out.velsurf_mag[-1].data
        
        thk_condition = (thk_obs > 0) & (thk_obs < 1e20)
        vel_condition = (vel_obs > 5) & (slidingco == slidingco[0,0]) & (mask == 1)

        vel_areas.append(vel_condition.sum())
        n_obs.append(thk_condition.sum())

        if vel_condition.any():
            mean_vel_obs.append(np.mean(vel_obs[vel_condition]))
            mean_vel_mod.append(np.mean(vel_mod[vel_condition]))
        else:
            mean_vel_obs.append(np.nan)
            mean_vel_mod.append(np.nan)

        if thk_condition.any():
            mean_thk_obs.append(np.mean(thk_obs[thk_condition]))
            mean_thk_mod.append(np.mean(thk_mod[thk_condition]))
            MAEs.append(np.mean(abs(thk_obs[thk_condition] - thk_mod[thk_condition])))
            MEs.append(np.mean(thk_obs[thk_condition] - thk_mod[thk_condition]))
        else:
            MAEs.append(np.nan)
            MEs.append(np.nan)
            mean_thk_obs.append(np.nan)
            mean_thk_mod.append(np.nan)

    out_df = pd.DataFrame(
        {
            'MAE': MAEs,
            'ME': MEs,
            'mean_thk_obs': mean_thk_obs,
            'mean_thk_mod': mean_thk_mod,
            'mean_vel_mod': mean_vel_mod,
            'mean_vel_obs': mean_vel_obs,
            'vel_areas': vel_areas/np.sum(vel_areas),
            'n_obs': n_obs/np.sum(n_obs)
        }
    )
        

    out_df.to_csv('validation_stats.csv')

    loss_MAE = (out_df['MAE'] * out_df['n_obs']).sum() / (out_df['mean_thk_obs'] * out_df['n_obs']).sum()
    loss_ME = (out_df['ME'] * out_df['n_obs']).sum() / (out_df['mean_thk_obs'] * out_df['n_obs']).sum()
    loss_vel = 1 - (out_df['mean_vel_mod'] * out_df['vel_areas']).sum() / (out_df['mean_vel_obs'] * out_df['vel_areas']).sum()
    
    print('LOSS MAE: ', loss_MAE, '*', L_MAE_f)
    print('LOSS ME: ', loss_ME, '*', L_ME_f)
    print('LOSS vel: ', loss_vel, '*', L_vel_f)
    

    return -(loss_MAE * L_MAE_f + abs(loss_ME) * L_ME_f + abs(loss_vel) * L_vel_f)


def create_script(glaciers, experiments):

    '''
    writes loop_script.sh with given parameters. this in turn runs igm over the given glaciers
    '''
    
    translate_dict = {
        'A': 'iflo_init_arrhenius',
        'theta': 'theta',
        'c': 'iflo_init_slidingco',
        'velMult': 'vel_mult'
    }

    params_string = '_'.join(['{}-{}'.format(x, experiments[x]) for x in experiments])
    with open('loop_script.sh', 'w') as file:
        file.write('#!/bin/bash\n\n')
        file.write('for param in ')
        file.write(' '.join(glaciers))
        file.write('\n')
        file.write('do\n')
        file.write('\techo "Processing $param"\n')
        file.write('\tpython3 igm_run ' +
                   '--lncd_input_file ../Input_data/$param_input.nc ' +
                   '--wncd_output_file ../Output_data/"$param"_output_{}.nc '.format(params_string) +
                   '--wts_output_file ../Output_data/"$param"_ts_{}.nc '.format(params_string))
        for key in experiments:
            file.write('--{} {} '.format(translate_dict[key], experiments[key]))

        file.write('\ndone\n')        
                             
if __name__ == '__main__':
    full_optimization(next_run, params, n_iter, old_log_paths = old_log_paths)
