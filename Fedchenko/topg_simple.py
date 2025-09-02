#!/usr/bin/env python3

# Published under the GNU GPL (Version 3), check at the LICENSE file

import warnings
import datetime, time
import tensorflow as tf
import numpy as np
from scipy import ndimage as nd
from igm.modules.utils import compute_divflux, compute_gradient_tf, str2bool

def params(parser):
    parser.add_argument(
        "--theta",
        type=float,
        default=0.1,
        help="fraction of bed updates used to update surface",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.,
        help="factor for scaling bed updates",
    )
    parser.add_argument(
        "--t_fr_update",
        type=int,
        default=100000,
        help="model time at which slidingco is updated (needs to be multiple of time_save)",
    )
    parser.add_argument(
        "--mask_buffer",
        type=int,
        default=0,
        help="n grid cells by which to buffer original mask",
    )
    parser.add_argument(
        "--crop_to_original",
        type=str2bool,
        default="True",
        help="if mask_buffer>0, crop final output to original mask?",
    )
    parser.add_argument(
        "--max_thk",
        type=float,
        default=2e3,
        help="upper limit on thickness",
    )    
    parser.add_argument(
        "--max_vel_ratio",
        type=float,
        default=0.8,
        help="absolute value of maximum relative velocity mismatch used when updating friction",
    )    
    parser.add_argument(
        "--ocean_bathymetry",
        type=float,
        default=-700.0,
        help="bed elevation in ocean (i.e. where surface elevation is zero)",
    )    
    parser.add_argument(
        "--topg_smb_array",
        type=list,
        default=[],
        help="Parameters for simple mass balance model (time, gradabl, gradacc, ela, accmax)",
    )

    
def initialize(params, state):

    state.tcomp_topg = []

    # calculate apparent mass balance from difference between smb and dhdt
    if not hasattr(state, 'amb'):
        if not hasattr(state, 'smb'):
            state.smbpar = np.array(params.smb_simple_array[1:]).astype(np.float32)
            # compute smb from glacier surface elevation and parameters
            state.smb = state.usurf - state.smbpar[: , 3]
            state.smb *= tf.where(tf.less(state.smb, 0), state.smbpar[:, 1], state.smbpar[:, 2])
            state.smb = tf.clip_by_value(state.smb, -100, state.smbpar[:, 4])

        state.amb = state.smb - state.dhdt

    # expecting zero amb outside mask
    state.amb = state.amb * state.icemask

    # check if amb sums to zero over domain
    net_amb = abs(tf.reduce_sum(state.amb))
    if net_amb > 0: 
        warnings.warn(
            """Apparent mass balance = {} != 0!
            For non-calving glaciers, this is often not desired as it violates mass conservation.""".format(net_amb)
        )

    # calculate velocity magnitude obs from u and v if not existant and if friction updates requested
    if (not hasattr(state, 'velsurf_magobs')) & (params.t_fr_update <= params.time_end):
        # u and v velocity must be provided
        assert ((hasattr(state, 'uvelsurfobs')) & (hasattr(state, 'vvelsurfobs')))
        state.velsurf_magobs = tf.sqrt(state.uvelsurfobs ** 2 + state.vvelsurfobs ** 2)

    # set buffer around glacier and extrapolate amb there (where amb negative)
    if (params.mask_buffer > 0):
        _create_buffer_with_smb(params, state)
    else:
        state.mask_buffer = tf.zeros_like(state.icemask)

    # set thk to zero and topg to usurf if not provided
    if not hasattr(state, 'thk'):
        if not hasattr(state, 'topg'):
            state.thk = tf.zeros_like(state.icemask)
            state.topg = state.usurf
        else:
            state.thk = state.usurf - state.topg
    elif not hasattr(state, 'topg'):
            state.topg = state.usurf - state.thk * state.icemask
    else:
        # no ice outside mask
        state.thk = state.thk * state.icemask
        # ensure consistency between thk and topg
        state.topg = state.usurf - state.thk
        
    # set minimum surface elevation to 0 and topg to bathymetry in ocean
    state.usurf = tf.maximum(state.usurf, 0)
    state.topg = tf.where(state.usurf<=0, params.ocean_bathymetry, state.topg)
    state.icemask = tf.where(state.usurf<=0, 0, state.icemask)
   
def update(params, state):
    
    if state.it>=0:

        if hasattr(state,'logger'):
            state.logger.info("Bed update equation at time : " + str(state.t.numpy()))

        state.tcomp_topg.append(time.time())

        # compute the divergence of the flux
        state.divflux = compute_divflux(state.ubar, state.vbar, state.thk, state.dx, state.dx)

        # compute dhdt mismatch (i.e. deviations from dhdt=0 when using apparent mass balance)
        state.dhdt = state.dt * (state.amb - state.divflux)

        # update thickness
        state.thk = tf.minimum(
            tf.maximum(
                state.thk + state.dhdt * params.beta * state.icemask,
                0),
            params.max_thk
        )

        # update bed topography 
        state.topg = tf.where((state.icemask == 1) & (state.usurf > 0), state.usurf - state.thk, state.topg)

        # update surface (but not in buffer)
        state.usurf = tf.maximum(
            tf.maximum(0, state.topg),
            state.usurf + state.dhdt * params.theta * params.beta * state.icemask * (1-state.mask_buffer)
        )

        # update friction; IMPORTANT: to hit t_fr_update, it needs to be a multiple of the saving time step
        if state.t>0 and state.t%params.t_fr_update==0:
            state.velsurf = tf.sqrt(state.uvelsurf ** 2 + state.vvelsurf ** 2)

            vel_mismatch = tf.maximum(
                tf.minimum(
                    (state.velsurf - state.velsurf_magobs)/state.velsurf_magobs,
                    params.max_vel_ratio),
                -params.max_vel_ratio
            )

            vel_mismatch = tf.where(
                (tf.math.is_nan(vel_mismatch)) | (state.velsurf_magobs < 1) | (state.icemask == 0),
                0,
                vel_mismatch
            )

            # apply friction update based on vel mismatch
            state.slidingco = vel_mismatch * state.slidingco + state.slidingco
            state.slidingco = tf.minimum(state.slidingco, 1)
            
        # final checks on thk and usurf
        state.thk = tf.where((state.icemask == 1) & (state.usurf > 0), state.usurf - state.topg, 0)
        state.usurf = tf.where((state.icemask == 0) & (state.topg < 0), 0, state.usurf)

        if (state.t == params.time_end) & params.crop_to_original:
            original_mask = state.icemask - state.mask_buffer
            state.thk = state.thk * original_mask
            state.topg = tf.where(original_mask == 1, state.topg, state.usurf)
            state.icemask = original_mask
            
        state.tcomp_topg[-1] -= time.time()
        state.tcomp_topg[-1] *= -1


def finalize(params, state):
    pass


def _create_buffer_with_smb(params, state):

    state.mask_buffer =  tf.convert_to_tensor(
        _internal_buffer(params.mask_buffer, 1-state.icemask.numpy()),
        dtype=tf.float32
    )
    
    try:
        # fits amb as function of elevation in ablation area
        amb_slope, amb_intercept = np.polyfit(state.usurf[state.amb<0], state.amb[state.amb<0], deg=1)
        amb_fit = amb_intercept + amb_slope * state.usurf
        # applies fit, but only negative values to not create mass
        state.amb = tf.where((amb_fit<0) & (state.mask_buffer == 1), amb_fit, state.amb)
    except:
        print('Setting apparent mass balance in buffer based on extrapolation not working! Setting to zero instead.')
        state.amb = tf.where((state.amb<0) & (state.mask_buffer == 1), 0, state.amb)

    state.icemask = state.icemask + state.mask_buffer


def _internal_buffer(bw, mask):

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
