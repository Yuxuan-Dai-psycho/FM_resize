#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:14:08 2022

@author: yuxuan dai
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import thingsvision.vision as vision
from scipy.io import loadmat
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, ttest_rel
from skimage.transform import rescale


def compute_FM(features_ori, resize_scale, layer_list, resize_type='ratio'):
    
    '''
    Downsaampling and combining feature maps for each layer.
     
    Parameters
    ----------
    features_ori: dict, key=['layer{i}'...]
        features
    
    resize_scale : list of resize scale(float)

    layer_list: list of layers
        
    resize_type : str, optional
        'fixed' or 'ratio'. The default is 'ratio'.

    Returns
    -------
    rescale_feature_maps: DataFrame

    '''
    
    if resize_type not in ['ratio', 'fixed']:
        raise ValueError("resize type must be 'ratio' or 'fixed'!")
    
    n_fig = list(features_ori.values())[0].shape[0]
    
    # downsaampling and combining feature maps
    re_fm_df = pd.DataFrame(columns=[f'rescale={resize}' for resize in resize_scale], 
                            index=[f'layer{i}' for i in layer_list])
    re_fm_df.applymap(lambda x: x if str(x) !='nan' else [])
        
    for resize in resize_scale:
    
        for layer_idx, features in features_ori.items():
            
            layer_feature = np.array([])
            
            for i in np.arange(n_fig):
                
                fig_feature = np.array([])
                tmp_features = features[i,:,:,:]
                
                if resize_type=='ratio':
                    
                    if np.around(tmp_features.shape[-1]*resize) > 0:
                        re_fm = rescale(tmp_features.T, scale=resize, order=1, multichannel=True)
                    
                    else:
                        min_resize = 1/tmp_features.shape[-1]
                        re_fm = rescale(tmp_features.T, scale=min_resize, order=1, multichannel=True)
               
                else: # resize_type=='fixed'
                
                    resize_ratio = resize/tmp_features.shape[-1]
                    
                    if resize > tmp_features.shape[-1]:
                        re_fm = tmp_features.T
                        
                    else:
                        re_fm = rescale(tmp_features.T, scale=resize_ratio, order=1, multichannel=True)
                    
                    
                if i == 0:
                    layer_feature = re_fm.flatten().reshape((1,-1))
                else:
                    layer_feature = np.concatenate((layer_feature, re_fm.flatten().reshape((1,-1))),
                                                    axis=0)
                
            re_fm_df[f'rescale={resize}'][layer_idx] = layer_feature
            
    return re_fm_df
            
            
def compute_rdv(fm_df, resize_scale, layer_list):
    '''
    compute RDVs

    Parameters
    ----------
    fm_df : DataFrame
    resize_scale : list
        *should match with fm_df*

    Returns
    -------
    fm_rdv : DataFrame

    '''
    
    fm_rdv = pd.DataFrame(columns=[f'rescale={resize}' for resize in resize_scale], 
                        index=[f'layer{i}' for i in layer_list])
    fm_rdv.applymap(lambda x: x if str(x) !='nan' else [])
    
    for rescale, col in fm_df.iteritems():
        
        for layer, layer_features in col.iteritems():
        
            # compute RDM
            features_rdm = vision.compute_rdm(layer_features, method='correlation')
            
            # feature_rdms --> RDV (RDV x n_features)
            feature_rdv = features_rdm[np.tril_indices(features_rdm.shape[0], k=-1)]
            
            fm_rdv[rescale][layer] = feature_rdv
        
    return fm_rdv


def compute_corr(fm_rdv, resize_scale, brain_rdm, layer_list):
    '''
    compute correlation scores of feature maps and brain RDM

    Parameters
    ----------
    fm_rdv : DataFrame
    resize_scale : list
        *should match with fm_df*
    brain_rdm : np.array
    layer_list : list

    Returns
    -------
    corr : DataFrame
    '''   

    corr = pd.DataFrame(columns=[f'rescale={resize}' for resize in resize_scale], 
                        index=[f'layer{i}' for i in layer_list])
    corr = corr.applymap(lambda x: x if str(x) !='nan' else [])
    
    for rescale, col in fm_rdv.iteritems():
        for layer, rdv in col.iteritems():
            layer_corr = np.zeros((1,brain_rdm.shape[2]))
            for sub_idx in np.arange(brain_rdm.shape[-1]):
                sub_corr = []
                for point in np.arange(brain_rdm.shape[2]):
                    hIT_rdm = brain_rdm[:,:,int(point),sub_idx]
                    hIT_rdm = hIT_rdm.squeeze()
                    flatten_hIT_rdm = hIT_rdm[np.tril_indices(hIT_rdm.shape[0], k=-1)]
                                            
                    # compute the pearson correlation
                    r, _ = pearsonr(rdv, flatten_hIT_rdm)
                    
                    sub_corr.append(r)

                layer_corr = np.concatenate((layer_corr, np.array(sub_corr)[np.newaxis,:]), axis=0)
                # layer_corr.shape=[n_sub, n_point] 
                # for meg data, n_point indicates timepoints, for fmri data, n_pint indicates ROI
            corr[rescale][layer] = np.delete(layer_corr,0,0)
       
    return corr