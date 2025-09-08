#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import datetime, time
import tensorflow as tf
import numpy as np
from scipy import ndimage as nd
from igm.modules.utils import compute_divflux, compute_gradient_tf

def params(parser):
    parser.add_argument(
        "--theta",
        type=float,
        default=0.1,
        help="fraction of bed updates used to update surface",
    )
    parser.add_argument(
        "--beta_0",
        type=float,
        default=1.,
        help="factor for scaling bed updates",
    )
    parser.add_argument(
        "--mask_thickness",
        type=int,
        default=1,
        help="force thickness to zero outside mask",
    )
    parser.add_argument(
        "--smb_outside_mask",
        type=float,
        default=0.0,
        help="prescribe a mass balance outside mask",
    )
    parser.add_argument(
        "--ablation_area_smooth",
        type=int,
        nargs='+',
        default=[2,1],
        help="parameters for gaussian smoothing of usurf in ablation area",
    )    
    parser.add_argument(
        "--debris_cover_smooth",
        type=int,
        nargs='+',
        default=[3,2],
        help="parameters for gaussian smoothing of usurf in debris covered areas",
    )    
    parser.add_argument(
        "--mask_buffer",
        type=int,
        default=0,
        help="n grid cells by which to buffer original mask",
    )    
    parser.add_argument(
        "--divflux_method",
        type=str,
        default="upwind",
        help="Compute the divergence of the flux using the upwind or centered method",
    )
    parser.add_argument(
        "--t_fr_update",
        type=float,
        default=1e36,
        help="time at which slidingco is updated",
    )
    parser.add_argument(
        "--it_save",
        type=float,
        default=1e36,
        help="save every nth iteration",
    )
    parser.add_argument(
        "--it_end",
        type=float,
        default=1e36,
        help="end inversion after n iterations",
    )
    parser.add_argument(
        "--t_end_min",
        type=float,
        default=1e36,
        help="simulate at least n years before stopping (if end defined by iterations)",
    )
    parser.add_argument(
        "--it_fr_update",
        type=float,
        default=1e36,
        help="iteration at which slidingco is updated",
    )

    
def initialize(params, state):

    state.tcomp_topg = []
    if hasattr(state, 'apparent_mass_balance'):
        state.smb = tf.convert_to_tensor(state.apparent_mass_balance)
        
    # smooth debris covered areas
    if hasattr(state, 'debris_mask') and not any(item == 0 for item in params.debris_cover_smooth):
        state.usurf = tf.where(
            state.debris_mask == 0,
            state.usurf,
            filter_tf(state.usurf, params.debris_cover_smooth[0], sigma = params.debris_cover_smooth[1])
        )
    
    # smooth surface
    if not any(item == 0 for item in params.ablation_area_smooth):
        state.usurf = tf.where(
            state.smb > 0,
            state.usurf,
            filter_tf(state.usurf, params.ablation_area_smooth[0], sigma = params.ablation_area_smooth[1])
        )


    if not hasattr(state, 'TermMask'):
        state.TermMask = state.mask * 0
        
    if (params.mask_buffer > 0) & (hasattr(state, 'mask_count')):
        
        _create_buffer_with_smb(params, state)

        state.mask = tf.where(state.TermMask == 0,
            state.mask + \
            tf.convert_to_tensor(
                internal_buffer(params.mask_buffer, 1-np.array(state.mask)),
                dtype=tf.float32
            ),
            0
        )
        state.no_theta_area = state.mask - state.mask_start

    else:
        state.no_theta_area = tf.zeros_like(state.thk)
        if not hasattr(state, 'mask_count'):
            state.mask_count = state.mask * 1

    # set mass balance outside mask
    state.smb = tf.where(state.mask == 1, state.smb, state.smb + params.smb_outside_mask)

    state.usurf = tf.where(state.TermMask == 1, 0, state.usurf)
    state.usurf = tf.maximum(state.usurf, 0)
    state.topg = tf.where(state.usurf<=0, -700.0, state.topg)
    state.topg = tf.maximum(state.topg, -700)
    
    if hasattr(state, 'TermType'):
        state.mask = state.mask * (-state.TermType + 1)
        state.thk = state.thk * state.mask

    if not hasattr(state, 'isMarine'):
        # if no marine mask provided, assume that friction updates should occur for all glaciers
        state.isMarine = state.mask * 1 
        
def update(params, state):
    
    if state.it>=0:

        if hasattr(state,'logger'):
            state.logger.info("Bed update equation at time : " + str(state.t.numpy()))

        state.tcomp_topg.append(time.time())

        # ramp-up beta
        beta = ((-20*params.beta_0)/(state.t+20)) + params.beta_0

        # compute the divergence of the flux
        state.divflux = compute_divflux(state.ubar, state.vbar, state.thk, state.dx, state.dx, method=params.divflux_method)

        # Forward Euler with projection to keep ice thickness non-negative
        state.dhdt = state.dt * (state.smb - state.divflux)
        if params.mask_thickness == 1:
            state.thk = tf.minimum(
                tf.maximum(
                    state.thk + state.dhdt * beta * state.mask,
                    0),
                2e3
            )
            
        else:
            state.thk = tf.minimum(tf.maximum(state.thk + state.dhdt * beta, 0), 2e3)

        # update bed topography 
        state.topg = tf.where(tf.logical_and(state.mask == 1, state.usurf > 1), state.usurf - state.thk, state.topg)
        state.topg = tf.maximum(state.topg, -700)
        
        # update surface
        state.usurf = tf.maximum(
            tf.maximum(0, state.topg),
            state.usurf + state.dhdt * params.theta * beta * state.mask * (1-state.no_theta_area)
        )

        # update friction
        if (state.t>0 and state.t%params.t_fr_update==0) or (state.it>0 and state.it%params.it_fr_update==0):
            state.velsurf = tf.norm(
                tf.concat([tf.expand_dims(state.uvelsurf, axis=-1),
                           tf.expand_dims(state.vvelsurf, axis=-1)], axis=2),
                axis=2,
            )
            vel_mismatch = tf.maximum(
                tf.minimum(
                    (state.velsurf - state.velsurf_magobs)/state.velsurf_magobs,
                    .8),
                -.8
            )
            vel_mismatch = tf.where(tf.math.is_nan(vel_mismatch) | (state.velsurf_magobs < 100) | (state.mask != 1) | (state.isMarine != 1), 0, vel_mismatch)
            state.slidingco = state.slidingco + vel_mismatch * state.slidingco
            state.slidingco = tf.minimum(state.slidingco, 1)
            

        # update thickness
        state.thk = tf.where((state.mask == 1) & (state.usurf > 1), state.usurf - state.topg, 0)
        state.usurf = tf.where((state.mask == 0) & (state.topg < 0), 0, state.usurf)
        
        state.tcomp_topg[-1] -= time.time()
        state.tcomp_topg[-1] *= -1

        if state.it%params.it_save == 0:
            state.saveresult = True
        else:
            state.saveresult = False

        if (state.it >= params.it_end):
            if (state.t >= params.t_end_min):
                # below breaks while loop in run_processes (common.py)
                params.time_end = state.t
            else:
                if state.it % 100 == 0:
                    print('Enough iterations reached, but t={} < t_end_min={}. Continuing..'.format(state.t.numpy(), params.t_end_min))


def finalize(params, state):
    pass



def filter_tf(raster, size, kernel_type = 'gauss', sigma=1):

    '''
    size refers to the number of grid points in both directions from the center,
    e.g. size = 1 gives a 3x3 kernel
    '''

    if kernel_type == 'gauss':
        x = tf.linspace(-size, size, 2*size+1)
        y = tf.linspace(-size, size, 2*size+1)
        x, y = tf.meshgrid(x, y)
        kernel = tf.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        kernel /= tf.reduce_sum(kernel)
        kernel = tf.cast(kernel, dtype = tf.float32)

    elif kernel_type == 'mean':
        kernel = tf.ones((2*size+1, 2*size+1))
        kernel /= tf.reduce_sum(kernel)
        
    elif kernel_type not in ['gauss', 'mean']:
        raise ValueError('No other filter types than gauss and mean are implemented, choose one of them')
    
    smoothed_raster = tf.nn.conv2d(input=tf.convert_to_tensor(raster[tf.newaxis, :,:,tf.newaxis], dtype = tf.float32),
                               filters=tf.reshape(kernel, (kernel.shape[0], kernel.shape[1], 1, 1)),
                               strides= 1,
                               padding='SAME')
    return smoothed_raster[0,:,:,0]


# buffer mask
def internal_buffer(bw, mask):

    #TODO: use tensorflow functions
    mask_iter = mask == 1
    mask_bw = ~mask_iter
    buffer = np.zeros_like(mask_iter)
    for i in range(bw):
        boundary_mask = mask_bw==0
        k = np.ones((3,3),dtype=int)
        boundary = nd.binary_dilation(boundary_mask==0, k) & boundary_mask
        mask_bw = np.where(boundary, 1, mask_bw)
    buffer = ((mask_bw + mask_iter)-1)
    return buffer


def _create_buffer_with_smb(params, state):
    state.no_theta_area = state.mask * 0
    for b in tf.unique(tf.reshape(state.mask_count, -1))[0]:
        if b == 0:
            continue
        one_glacier = tf.where(
            state.mask_count == b,
            1.0,
            0.0
        ) * state.mask

        if (tf.reduce_max(state.usurf[one_glacier == 1]) - tf.reduce_min(state.usurf[one_glacier == 1])) > .4e3:
            one_glacier_buffer = one_glacier + \
                tf.convert_to_tensor(
                    internal_buffer(3, 1-np.array(one_glacier)),
                    dtype=tf.float32
                )
            # do not extent buffer to surrounding surging glaciers
            only_margin = tf.maximum(0, one_glacier_buffer - state.mask)
            min_k_usurf = np.argsort(state.usurf.numpy().flatten() + (9999.0*(1-one_glacier.numpy())).flatten())[:5]

            lowest_usurf = np.zeros_like(state.mask.numpy().flatten())
            lowest_usurf[min_k_usurf] = 1
            lowest_usurf = lowest_usurf.reshape(state.usurf.numpy().shape)

            lowest_buffer = tf.convert_to_tensor(
                internal_buffer(3, 1-np.array(lowest_usurf)),
                dtype=tf.float32
            )

            state.no_theta_area += tf.maximum(0, lowest_buffer - state.mask)
        if np.array(state.smb[(state.smb*one_glacier)<0]).size:
            one_glacier_buffer = one_glacier + \
                tf.convert_to_tensor(
                internal_buffer(params.mask_buffer, 1-np.array(one_glacier)),
                dtype=tf.float32
            )
            only_margin = tf.maximum(0, one_glacier_buffer - state.mask)
            if len(np.unique(state.smb[(state.smb * one_glacier)<0].numpy())) > 2:
                smb_slope, smb_intercept = np.polyfit(state.usurf[(state.smb*one_glacier)<0], state.smb[(state.smb*one_glacier)<0], deg=1)
                smb_fit = smb_intercept + smb_slope * state.usurf
                state.smb = tf.where(tf.logical_and(smb_fit<0, only_margin == 1), smb_fit, state.smb)

    state.no_theta_area = tf.minimum(1, state.no_theta_area)
