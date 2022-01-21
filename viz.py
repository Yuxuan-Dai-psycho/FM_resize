#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jan 19 11:38:07 2022

@author: yuxuan dai
"""

import numpy as np
import os
from os.path import join as pjoin
from matplotlib import pyplot as plt
import thingsvision.vision as vision
from scipy.io import loadmat
import h5py
import pandas as pd


# data transform function
def viz_df_trans(ori_df, target_col, point_list, set_idx, re_type):

    target_df = pd.DataFrame(columns=target_col)
    for i in np.arange(ori_df.shape[0]):
        for j in np.arange(ori_df.shape[1]):
            single_corr = ori_df.iloc[i,j]
            flat_corr = ori_df.iloc[i,j].flatten()

            # subject columns
            sub_col = np.repeat(np.arange(1,single_corr.shape[0]+1).reshape(-1,1), single_corr.shape[1], axis=1)
            sub_col = sub_col.flatten()

            cond_dict = {target_col[0]: flat_corr,
                        target_col[1]: [f'dataset{set_idx}']*flat_corr.size,
                        target_col[2]: [list(ori_df.index)[i]]*flat_corr.size,
                        target_col[3]: [re_type]*flat_corr.size,
                        target_col[4]: [list(ori_df.columns)[j][8:]]*flat_corr.size,
                        target_col[5]: sub_col,
                        target_col[6]: list(point_list)*single_corr.shape[0]}

            target_df = pd.concat((target_df, pd.DataFrame(cond_dict)), ignore_index=True)

    return target_df


# noise ceiling computation

# load rdm
def load_rdm(modality, n_fig):

  dataset_root = '/LOCAL/ydai/workingdir/dataset'

  if modality == 'meg':
  # load meg rdm
    if n_fig == 92:
      # for 92 dataset
      meg_rdm = h5py.File(pjoin(dataset_root, f'{n_fig}_object_imageset/MEG_RDMs/MEG_decoding_RDMs.mat'), 'r')
      sess1_hIT_rdm = meg_rdm['MEG_decoding_RDMs'][:,:,:,0,:]
      sess2_hIT_rdm = meg_rdm['MEG_decoding_RDMs'][:,:,:,1,:]
      target_rdm = np.mean(meg_rdm['MEG_decoding_RDMs'], axis=3)
      # target_rdm.shape=[92,92,n_timepoints,n_sub]

      # if on session-level
      # target_rdm_sess = {'sess1': sess1_hIT_rdm,
      #                    'sess2': sess2_hIT_rdm}
      
    elif n_fig == 118:
      file_name = pjoin(dataset_root, f'{n_fig}_object_imageset/MEG_RDMs/MEG_decoding_RDMs.mat')
      meg_rdm = loadmat(file_name)
      brain_rdm = meg_rdm['MEG_decoding_RDMs']
      target_rdm = brain_rdm.squeeze().T
      # target_rdm shape: (118, 118, 1101，n_sub)

  elif modality == 'fmri':
    # load fmri rdm
    fmriroi_root = pjoin(dataset_root, f'{n_fig}_object_imageset', 'fmri_roidata_new')
    evc_fn = f'{n_fig}_fmri_evc_raw_new250_single.mat'
    hvc_fn = f'{n_fig}_fmri_hvc_raw_new250_single.mat'
    evc_data = loadmat(os.path.join(fmriroi_root, evc_fn))['data']
    hvc_data = loadmat(os.path.join(fmriroi_root, hvc_fn))['data']

    sub_list = ['{0:0>2d}'.format(i+1) for i in np.arange(evc_data.shape[1])]

    evc_rdm = np.zeros((n_fig, n_fig, len(sub_list)))
    hvc_rdm = np.zeros((n_fig, n_fig, len(sub_list)))
    for i in np.arange(len(sub_list)):
        evc_rdm[:,:,i] = vision.compute_rdm(evc_data[0,i].T, method='correlation')
        hvc_rdm[:,:,i] = vision.compute_rdm(hvc_data[0,i].T, method='correlation')
    
    # combine evc and hvc rdm
    target_rdm = np.concatenate((evc_rdm[:,:,np.newaxis,:], hvc_rdm[:,:,np.newaxis,:]), axis=2)
    # the 3rd dim of target rdm means ROI index, 0=early visual cortex, 1=higher visual cortex
  
  elif modality == 'behavior':   
  # load behavior rdm
    behavior_root = pjoin(dataset_root, f'{n_fig}_object_imageset/behavior')
    behav_fn = f'{n_fig}_behavior_rdm_single.mat'
    target_rdm = loadmat(pjoin(behavior_root, behav_fn))[f'RDM{n_fig}_arrange']

  return target_rdm

def to_rdv(rdm):

  return rdm[np.tril_indices(rdm.shape[0], k=-1)]

def splitter(rdv):
  """split the data NFold, leaving out one element of b (first dim) each time.
  Returns the mean over the training split (axis=1) and the test split."""
  nsplit = rdv.shape[1]
  allind = np.arange(nsplit)
  for testind in allind:
    trainind = np.setdiff1d(allind, testind)
    yield np.mean(rdv[:,trainind], axis=1), rdv[:, testind]

def ceiling(rdv):

  group_level_rdv = np.mean(rdv, axis=1)

  ceils = []
  for _, test in splitter(rdv):

    r = np.corrcoef(group_level_rdv, test)[0][1]
    rz = np.arctanh(r)

    ceils.append(rz)

  upper_ceil = np.tanh(np.mean(ceils))

  return upper_ceil

def compute_noise_ceiling(modality, set_idx):
  target_rdm = load_rdm(modality, set_idx)
  rdv = to_rdv(target_rdm)
  
  if modality == 'fmri':
    evc_rdv = rdv[:,0,:].squeeze()
    hvc_rdv = rdv[:,1,:].squeeze()
    noise_ceiling = [ceiling(evc_rdv), ceiling(hvc_rdv)]
  
  elif modality == 'meg':
    rdv = np.mean(rdv, axis=1)
    noise_ceiling = ceiling(rdv)

  elif modality == 'behavior':
    noise_ceiling = ceiling(rdv)

  return noise_ceiling

    