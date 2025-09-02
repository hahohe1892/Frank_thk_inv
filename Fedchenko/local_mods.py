#!/usr/bin/env python3

# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from netCDF4 import Dataset
from igm.modules.utils import compute_gradient_tf
from scipy.ndimage import convolve
from scipy.interpolate import griddata
import numpy as np
from topg_update import internal_buffer, filter_tf

def params(parser):
    parser.add_argument(
        "--vel_mult",
        type=float,
        default=1.0,
        help="Constant to multiply vel obs with",
    )
def initialize(params, state):

    # create term mask
    dx = round(abs(state.x.numpy()[0] - state.x.numpy()[1]), 0)
    state.TermMask = tf.where(state.usurf < 5, 1, state.TermMask)
    _clean_masks(params, state)
    state.icemask = tf.cast(state.icemask, tf.float32)
    state.mask = state.icemask

    ocean_buffer = tf.convert_to_tensor(internal_buffer(1, 1-state.TermMask.numpy()), dtype = tf.float32)
    marine_touch = tf.unique(tf.reshape(ocean_buffer * state.mask_count, -1))[0]
    marine_touch = marine_touch[marine_touch != 0]
    state.isMarine = tf.convert_to_tensor(np.isin(state.mask_count, marine_touch), dtype = tf.float32)
    
    # create empty variable needed for plotting (?)
    state.velsurf = tf.zeros_like(state.thk)

    state.velsurf_magobs = tf.sqrt(state.uvelsurfobs**2 + state.vvelsurfobs**2)

    # interpolate thk where neighboring glaciers touch since consensus thk is bad there
    state.thk = tf.convert_to_tensor(
        _smooth_and_interpolate_boundaries_consensus(state.thk.numpy(), state.mask_count.numpy(), state.mask.numpy()),
        dtype = tf.float32
    )
    
    state.velsurf_magobs = state.velsurf_magobs * params.vel_mult

    state.usurf = tf.where(state.TermMask == 1, 0.0, state.usurf)
    state.topg = tf.where(state.usurf <= 0, -700.0, state.topg)

    
def update(params, state):
    pass

def finalize(params, state):
    pass


def _clean_masks(params, state):

    from scipy.ndimage import label

    dt = np.maximum(state.icemask.numpy() - state.TermMask.numpy(), 0)

    # identify pixels that form a narrow link between parts of mask
    kernel_x = np.array([[0,0,0],[1,0,1],[0,0,0]])
    kernel_y = np.array([[0,1,0],[0,0,0],[0,1,0]])
    dt_convolved_x =convolve(dt, kernel_x)
    dt_convolved_y =convolve(dt, kernel_y)
    dt_clean = np.where(~(((dt_convolved_x == 2) & (dt_convolved_y == 0)) | ((dt_convolved_y == 2) & (dt_convolved_x == 0))) & (dt == 1), 1, 0)

    # make buffer around TermMask
    kernel_Term = np.array([[0,1,0],[1,0,1],[0,1,0]])
    Term_convolved = convolve(state.TermMask.numpy(), kernel_Term)
    buffer_around_TermMask = ((Term_convolved<4) & (Term_convolved > 0) & (state.TermMask.numpy() == 0)).astype('int')

    # only narrow links adjacent to TermMask should be set to zero
    dt_clean_front = np.where((buffer_around_TermMask + (dt-dt_clean))!=2, dt, 0)

    # save what was removed from mask
    mask_remains = dt - dt_clean_front

    # identify clusters in mask
    labeled_raster, num_features = label(dt_clean_front)
    labeled_raster = labeled_raster.astype('float')
    labeled_raster[state.icemask.numpy() == 0] = np.nan

    # only keep largest cluster, but do that for each basin seperately
    out_mask = np.zeros_like(labeled_raster)
    elements_mask = np.zeros_like(labeled_raster)
    for b in np.unique(state.mask_count):
        if b == 0:
            continue
        one_glacier = np.where(state.mask_count == b, 1, 0)
        if np.max(one_glacier * state.isMarine) > 0:
            one_glacier_labeled = one_glacier * labeled_raster
            elements = np.unique(one_glacier_labeled)
            elements = elements[(elements != 0) & ~(np.isnan(elements))]
            one_glacier_labeled[one_glacier_labeled == 0] = np.nan
            count = [len(np.nonzero(one_glacier_labeled == i)[0]) for i in elements]
            if len(count) == 0:
                continue
            max_element = elements[count == np.max(count)][0]
            out_mask = np.where(one_glacier_labeled == max_element, 1, out_mask)

            # identify removed patches next to TermMask
            elements_at_front = np.unique(one_glacier_labeled[buffer_around_TermMask == 1])
            elements_at_front = elements_at_front[(elements_at_front != max_element) & (~np.isnan(elements_at_front))]
            elements_mask = np.where(np.isin(one_glacier_labeled, elements_at_front), 1, elements_mask)
        else:
            out_mask = np.where((one_glacier*dt_clean_front) == 1, 1, out_mask)
            elements_mask = np.where((one_glacier*dt_clean_front) == 1, 0, elements_mask)
    
    # removed patches close to front and mask_remains goes to TermMask
    clean_Term = np.minimum(1, mask_remains + state.TermMask.numpy() + elements_mask)

    # identify isolated pixels in TermMask surrounded by ice
    donout_kernel = np.array([[1,1,1],[1,0,1], [1,1,1]])
    out_mask_convolved = convolve(out_mask, donout_kernel)
    out_Term = np.where((out_mask_convolved == 8) & (clean_Term == 1), 0, clean_Term)

    # add removed pixels to mask
    out_mask = out_mask + (clean_Term - out_Term)

    # identify isolated mask = 0 surrounded by TermMask
    out_Term_convolved = convolve(out_Term, donout_kernel)
    out_Term = np.where((out_Term_convolved == 8) & (out_mask == 0), 1, out_Term)
    out_mask = np.where((out_Term_convolved == 8) & (out_mask == 0), 0, out_mask)
    state.icemask = tf.Variable(out_mask)
    state.TermMask = tf.Variable(out_Term)


def _calc_init_slidingco(params, state):
    '''
    calculate sliding co based on perfect plasticity assumption
    '''
    dx = abs(state.x[0] - state.x[1])
    
    slope_x, slope_y = compute_gradient_tf(state.usurf, dx, dx)
    
    slope = tf.norm(
                tf.concat([tf.expand_dims(slope_x, axis=-1),
                           tf.expand_dims(slope_y, axis=-1)], axis=2),
                axis=2,
            )

    criterion = (state.velsurf_magobs > 1) & (state.thkobs > 0)

    # set slope threshold
    slope = tf.maximum(slope, 0.014)
    # smooth slope
    kernel = np.array([[1,1,1],[1,1,1], [1,1,1]])
    slope = convolve(slope, kernel) / 9

    # calc basal shear stress at obs locations
    tau_b = state.thkobs * 900 * 9.8 * tf.tan(slope)

    # normalize usurf and use as a proxy for sliding contribution
    in_mask = state.mask == 1
    b = (state.usurf - tf.reduce_min(state.usurf[in_mask])) \
        / (tf.reduce_max(state.usurf[in_mask]) - tf.reduce_min(state.usurf[in_mask]))

    b = tf.maximum(tf.minimum(b, 0.9), 0.1)
    
    # calculate tau_c based on Weertman sliding law with m=3
    tau_c = (state.velsurf_magobs * b / (tau_b * 1e-9)) ** (-1/3) # use units m/yr for vel and MPa for tau

    # ((u*b)/alpha)**(-1/3) is highly correlated with tau_c (see print statement)
    #print(np.corrcoef(((state.velsurf_magobs[criterion] * (1 - b[criterion]))/slope[criterion])**(-1/3), tau_c[criterion]))

    # use this to infer tau_c away from obs locations
    fit = np.polyfit(((state.velsurf_magobs[criterion] * b[criterion])/slope[criterion])**(-1/3), tau_c[criterion], 1)
    
    tau_c = tf.where(
        (tau_c == 0.0) | tf.math.is_nan(tau_c),
        fit[1] + ((state.velsurf_magobs * b) / slope)**(-1/3) * fit[0],
        tau_c
    )

    mean_tauc = tf.reduce_mean(tau_c[tau_c < 0.5])
    tau_c = tf.where((state.mask == 0) | (state.velsurf_magobs < 1), mean_tauc, tau_c)
    tau_c = tf.where(tau_c > 0.5, mean_tauc, tau_c)

    #smooth tau_c
    for i in range(3):
        tau_c = convolve(tau_c, kernel) / 9

    return tf.convert_to_tensor(tau_c)


def _smooth_and_interpolate_boundaries_consensus(thk, mask_count, mask):
    buffer_width = 10
    reduce_buffer = True
    while reduce_buffer:
        invalid = thk < -1e20
        for b in np.unique(mask_count):
            if b == 0:
                continue
            one_glacier = np.where(mask_count == b, 1.0, 0.0)
            one_glacier_buffer = internal_buffer(buffer_width, 1 - one_glacier)
            invalid = invalid | ((one_glacier_buffer * mask_count) != 0)

        # if more than 30% of glacier area is invalid, reduce buffer
        if (invalid.sum() > (0.3 * mask.sum())) & (buffer_width > 1):
            reduce_buffer = True
            buffer_width -= 1
        else:
            reduce_buffer = False

    glacier_counts = np.unique(mask_count, return_counts = True)
    not_nan_glacier_counts = np.unique(mask_count[~invalid], return_counts = True)

    # check which glaciers are entirely marked as invalid
    lost_glaciers = ~np.isin(glacier_counts[0], not_nan_glacier_counts[0])
    # check which glaciers have invalid area larger than 30%
    too_small_glaciers = list(not_nan_glacier_counts[0][(glacier_counts[1][~lost_glaciers]/not_nan_glacier_counts[1]) > 3])
    redo_glaciers = too_small_glaciers + list(glacier_counts[0][lost_glaciers])
    new_invalid = np.zeros_like(thk)
    # redo buffer calculation for redo_glaciers
    for g in redo_glaciers:
        one_glacier = np.where(mask_count == g, 1.0, 0.0)
        increase_buffer = True
        buffer_width = 1
        while increase_buffer:
            one_glacier_buffer = np.maximum(internal_buffer(buffer_width, one_glacier) - internal_buffer(buffer_width, 1-mask), 0)
            if (one_glacier_buffer.sum() < (0.3 * one_glacier.sum())):
                buffer_width += 1
            else:
                one_glacier_buffer = np.maximum(internal_buffer(buffer_width - 1, one_glacier) - internal_buffer(buffer_width - 1, mask), 0)
                increase_buffer = False
        new_invalid += one_glacier_buffer

    invalid[np.isin(mask_count, redo_glaciers)] = False
    invalid[new_invalid == 1] = True
    valid = ~invalid    
    
    i,j = np.indices(thk.shape)
    valid_points = np.array([i[valid], j[valid]]).T
    invalid_points = np.array([i[invalid], j[invalid]]).T

    for i in range(3):
        thk[invalid] = np.nan
        valid_values = thk[valid]
        thk[invalid] = griddata(valid_points, valid_values, invalid_points, method='linear')

        # heavily smooth consensus thickness
        thk = filter_tf(thk, 3, sigma = 5).numpy()

    return thk
