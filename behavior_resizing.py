#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jan 12 14:41:34 2022

@author: yuxuan dai
"""

import numpy as np
import os
from os.path import join as pjoin
import thingsvision.vision as vision
from scipy.io import loadmat
import time
import pandas as pd
from utlis import compute_FM, compute_rdv, compute_corr

#%% define variables
layer_list = np.arange(1, 6)
datasets_n = [92, 118]
fm_root = '/LOCAL/ydai/workingdir/feature_maps'
dataset_root = '/LOCAL/ydai/workingdir/dataset'
save_root = '/LOCAL/ydai/workingdir/resizing_results/behavior'
if not os.path.isdir(save_root):
    os.makedirs(save_root)

ratio_resize_scale = [0.05, 0.1, 0.2, 0.25, 0.5, 1]
fixed_resize_scale = [np.square(i) for i in np.arange(1,7)]

#%% resizing

for n_fig in datasets_n:
    # read extant feature maps 
    feature_dir = pjoin(fm_root, f'{n_fig}_vgg16_resmat.mat')
    features = loadmat(feature_dir)
    features = features[list(features.keys())[-1]]

    features_ori = {}
    for i in layer_list:
        features_ori[f'layer{i}'] = features[0][i-1].T

    sTime = time.time()

    # ratio_rescale 
    ratio_fm_df = compute_FM(features_ori, ratio_resize_scale, layer_list, 'ratio')    
    ratio_fm_rdv = compute_rdv(ratio_fm_df, ratio_resize_scale, layer_list)

    # fixed_rescale 
    fixed_fm_df = compute_FM(features_ori, fixed_resize_scale, layer_list, 'fixed')    
    fixed_fm_rdv = compute_rdv(fixed_fm_df, fixed_resize_scale, layer_list)

    eTime = time.time()
    time_cost = eTime - sTime
    print('dataset{} feature maps resizing spend {:.0f}h {:.0f}min {:.2f}sec'.format(n_fig, time_cost//3600, (time_cost%3600)//60, time_cost%60))

    # load behavior rdm
    behavior_root = pjoin(dataset_root, f'{n_fig}_object_imageset/behavior')
    behav_fn = f'{n_fig}_behavior_rdm_single.mat'
    behav_rdm = loadmat(pjoin(behavior_root, behav_fn))[f'RDM{n_fig}_arrange']

    # behavior_rdm transform
    target_rdm = behav_rdm[:,:,np.newaxis,:]
    # shape should be [n_fig, n_fig, 1, n_sub]

    sTime = time.time()
    
    # compute correlation between resized feature maps and meg RDM
    ratio_corr = compute_corr(ratio_fm_rdv, ratio_resize_scale, target_rdm, layer_list)
    fixed_corr = compute_corr(fixed_fm_rdv, fixed_resize_scale, target_rdm, layer_list)

    eTime = time.time()
    time_cost = eTime - sTime
    print('dataset{} correlation computation spend {:.0f}h {:.0f}min {:.2f}sec'.format(n_fig, time_cost//3600, (time_cost%3600)//60, time_cost%60))

    # save the data
    ratio_corr.to_pickle(pjoin(save_root, f'dataset{n_fig}_ratio_corr.pkl'))
    fixed_corr.to_pickle(pjoin(save_root, f'dataset{n_fig}_fixed_corr.pkl'))
