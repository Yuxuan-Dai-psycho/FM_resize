#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 08:38:42 2021

@author: daiyuxuan
"""

import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torchvision import models, transforms
import thingsvision.vision as vision
from scipy.io import loadmat
import h5py
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, ttest_rel
from scipy.interpolate import interp2d
from itertools import permutations
import re
import time


#%% feature extracting

model_name = 'vgg19'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, transforms = vision.load_model(model_name, pretrained=True, device=device)
# vision.show_model(model, model_name)
conv_idx = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
module_list = [f'features.{i}' for i in conv_idx]

for group_idx in np.arange(1, 3):

    root = f'/nfs/s2/userhome/daiyuxuan/workingdir/MPI/dataset/84_images/images/images84_{group_idx}_square'
    
    for module_name in module_list:
        outpath = f'./dataset/84_images/group{group_idx}/{model_name}/{module_name}/features'
        
        dl = vision.load_dl(root=root, out_path=outpath, batch_size=64, transforms=transforms)
        features, targets, probas = vision.extract_features(model, dl, module_name, batch_size=64, flatten_acts=False, device=device, return_probabilities=True)
        
        vision.save_features(features, outpath, '.npy')
        print(f'complete group{group_idx} {module_name } of 84_imageset')
        
for set_idx in [92, 118]:
    
    root = f'./dataset/{set_idx}_object_imageset/{set_idx}images'
    
    for module_name in module_list:
        outpath = os.path.join(f'./dataset/{set_idx}_object_imageset', f'{model_name}/{module_name}/features')
        
        dl = vision.load_dl(root=root, out_path=outpath, batch_size=64, transforms=transforms)
        features, targets, probas = vision.extract_features(model, dl, module_name, batch_size=64, flatten_acts=False, device=device, return_probabilities=True)
        
        vision.save_features(features, outpath, '.npy')
        print(f'complete {module_name } of {set_idx}_imageset')


#%% define functions

def extract_features(feature_root, model_name, layer_list):
    
    feature_ori = {}
    
    for layer_idx in layer_list:
        feature_pth = os.path.join(feature_root, f'{model_name}/features.{layer_idx}/features')
        features = vision.slices2tensor(feature_pth, 'features.txt')
        feature_ori[f'layer{layer_idx}'] = features
        print(f'layer{layer_idx} extracted')
        
    return feature_ori

def compute_FM(features_ori, resize_scale, scale_type='ratio'):
    
    '''
    Downsaampling and combining feature maps for each layer.
     
    Parameters
    ----------
    features_ori: dict, key=['layer{i}'...]
        features
    
    resize_scale : list of (resize_W, resize_H)
        W, H can be reduced ratio or fixed reduction scale
        
    scale_type : str, optional
        'fixed' or 'ratio'. The default is 'ratio'.

    Returns
    -------
    rescale_feature_maps: DataFrame

    '''
    
    xrange = lambda x: np.linspace(0, 1, x)
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
                
                for j in np.arange(tmp_features.shape[0]):
                    
                    single_feature = tmp_features[j,:,:]
                
                    # downsampling
                    W, H = single_feature.shape[:2]
                    
                    if scale_type == 'ratio':
                        new_W, new_H = (int(W/resize[0]), int(H/resize[1]))
                    else:
                        new_W, new_H = (int(resize[0]), int(resize[1]))
                    
                    f = interp2d(xrange(W), xrange(H), single_feature, kind="linear")
                    resize_feature = f(xrange(new_W), xrange(new_H))
                    
                    # flatten
                    fig_feature = np.append(fig_feature, 
                                              resize_feature.reshape((1,resize_feature.size)))
            
                if i == 0:
                    layer_feature = fig_feature.reshape((1, fig_feature.size))
                else:
                    layer_feature =np.concatenate((layer_feature, fig_feature.reshape((1, fig_feature.size))), axis=0)
                
            re_fm_df[f'rescale={resize}'][layer_idx] = layer_feature
            
    return re_fm_df
            
def compute_rdv(fm_df, resize_scale):
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

def compute_corr(fm_rdv, resize_scale, brain_rdm):
    
    time_corr = pd.DataFrame(columns=[f'rescale={resize}' for resize in resize_scale], 
                        index=[f'layer{i}' for i in layer_list])
    time_corr = time_corr.applymap(lambda x: x if str(x) !='nan' else [])
    
    for rescale, col in fm_rdv.iteritems():
        
        for layer, rdv in col.iteritems():
        
            for time_point in np.arange(brain_rdm.shape[2]):
                hIT_rdm = brain_rdm[:,:,int(time_point),:]
                flatten_hIT_rdm = np.mean(hIT_rdm[np.tril_indices(hIT_rdm.shape[0], k=-1)], axis=-1)
                                          
                # compute the pearson correlation
                r, _ = pearsonr(rdv, flatten_hIT_rdm)
                
                time_corr[rescale][layer].append(r)
       
    return time_corr
         
def plot_time_corr(time_corr, ori_corr, plot_type='separate'):
    
    if plot_type == 'separate':
        
        # plot coef separately
        for layer_idx, row in time_corr.iterrows():
            plt.figure(figsize=(8,5))
            plt.plot(ori_corr[layer_idx], linestyle='--', color='k')
            for label, coefs in row.iteritems():
                plt.plot(coefs, label=label)
            plt.xticks(ticks=np.arange(0, 131, 30), labels=[f'{i}ms' for i in np.arange(-200, 1100, 300)])
            plt.ylabel('pearson r')
            plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, fontsize=8)
            plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
            plt.title(layer_idx)
            
    elif plot_type=='combined':
        
        # plot coefs as subplots
        plt.figure()    
        for i, (layer_idx, row) in enumerate(time_corr.iterrows()):
            
            plt.subplot(2,3,i+1)
            plt.plot(ori_corr[layer_idx], linestyle='--', color='k')
            
            for label, coefs in row.iteritems():
                plt.plot(coefs, label=label)
            plt.xticks(ticks=np.arange(0, 61, 10), labels=[f'{i}ms' for i in np.arange(-100, 501, 100)])
            plt.ylim(top=0.3)
            plt.ylabel('pearson r')
            plt.title(layer_idx)
            
            if i==4:
                plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, fontsize=8)

def compute_eval_score(time_corr, ori_corr):
    
    eval_score = {}
    
    for layer_idx, row in time_corr.iterrows():
        
        scores = {'score_diff':[], 'n_tp':[], 'resize':[], 'corr':[]}
        
        for resize, corr in row.iteritems():
            
            if np.where(np.array(corr) > np.array(ori_corr[layer_idx]))[0].size:
                
                score_diff = np.sum(np.array(corr) - np.array(ori_corr[layer_idx]))
                n_tp = np.where(np.array(corr) > np.array(ori_corr[layer_idx]))[0].size
                
                # if score_diff > 15:
                #     print(f'{layer_idx} {df_type[i]} dwnsp of {resize} is {score_diff} higher at {n_tp} timepoints')
                #     plt.figure()
                #     plt.plot(corr[layer_idx], label=resize)
                #     plt.plot(ori_corr[layer_idx], linestyle='--', label='control')
                #     plt.title(layer_idx)
                #     plt.legend()
                
                scores['score_diff'].append(score_diff)
                scores['resize'].append(resize)
                scores['n_tp'].append(n_tp)
                scores['corr'].append(corr)
                
        scores = pd.DataFrame(scores)       
        eval_score[layer_idx] = scores.sort_values(by='score_diff', 
                                                   ascending=False, ignore_index=True)
    
    return eval_score

def plot_eval_score(eval_score, ori_corr):
    
    for dwnsp_type, layer_scores in eval_score.items():
    
        plt.figure()
        plt.suptitle(f'{dwnsp_type} downsampling')
        
        for i, (layer_idx, scores) in enumerate(layer_scores.items()):
            
            plt.subplot(2,3,i+1)
            plt.plot(ori_corr[layer_idx], label='original', linestyle='--', color='k')
            
            for idx in np.arange(0,5):
                plt.plot(scores.iloc[idx,-1], label=scores.iloc[idx,-2])
            
            # plt.xticks(ticks=np.arange(0, 1301, 300), labels=[f'{i}ms' for i in np.arange(-100, 1201, 300)])
            plt.xticks(ticks=np.append(np.arange(0, 1101, 300), 1100), 
                        labels=[f'{i}ms' for i in np.append(np.arange(-100, 1001, 300), 1000)])
            plt.ylim(top=0.35)
            plt.ylabel('pearson r')
            plt.title(layer_idx)
            plt.legend(loc=0, fontsize=6)

#%% extract features

sTime = time.time()

layer_list = [2, 7, 14, 21, 28]
# layer_list = [3, 10, 17, 24, 27, 37]
feature_root = './dataset/118_object_imageset'
features_ori = extract_features(feature_root, 'vgg16', layer_list)
    
# plt.figure()
# plt.suptitle('layer3 feature maps')
# for i in np.arange(features_ori['layer3'].shape[1]):
#     plt.subplot(8,8,i+1)
#     plt.imshow(features_ori['layer3'][1,i,:,:])
#     plt.axis('off')
# plt.tight_layout()

#%% ratio_rescale  

# sTime = time.time()
ratio_resize_scale = [(i,i) for i in range(1,8)]
ratio_resize_scale.extend(list(permutations(range(1,8),2)))

ratio_fm_df = compute_FM(features_ori, ratio_resize_scale, scale_type='ratio')
    
ratio_fm_rdv = compute_rdv(ratio_fm_df, ratio_resize_scale)

#%% fixed rescale
# resize_scale = [(i,i) for i in range(2,11)]
# resize_scale.extend(list(permutations(range(2,11),2)))

# fixed_fm_df = compute_FM(features_ori, resize_scale, scale_type='fixed')
    
# fixed_fm_rdv = compute_rdv(fixed_fm_df, resize_scale)

eTime = time.time()
time_cost = eTime - sTime
print('spend {:.0f}h {:.0f}min {:.2f}sec'.format(time_cost//3600, (time_cost%3600)//60, time_cost%60))

#%% vgg_rdm hIT comparsion

sTime = time.time()

# covert hIT
# 84 dataset
# meg_rdm = loadmat('/nfs/s2/userhome/daiyuxuan/workingdir/MPI/dataset/84_images/meg_rdm.mat')
# group2_rdm = meg_rdm['RDM84_MEG_group2']

# meg_rdm = loadmat('/nfs/s2/userhome/daiyuxuan/workingdir/MPI/dataset/118_object_imageset/MEG_RDMs/MEG_decoding_RDMs.mat')
meg_rdm = h5py.File('/nfs/s2/userhome/daiyuxuan/workingdir/MPI/dataset/92_object_imageset/MEG_decoding_RDMs.mat', 'r')

# for 92 dataset
# sess1_hIT_rdm = meg_rdm['MEG_decoding_RDMs'][:,:,:,0,:]
# sess2_hIT_rdm = meg_rdm['MEG_decoding_RDMs'][:,:,:,1,:]
mean_hIT_rdm = np.mean(meg_rdm['MEG_decoding_RDMs'], axis=3)

# for 118 dataset
# mean_hIT_rdm = meg_rdm['MEG_decoding_RDMs'][:,0,:,:,:].T 


# average meg edm from 0ms to 500ms
# hIT_rdm = np.mean(group2_rdm[:,:,int(132/1100*100):int(132/1100*600),:], axis=2)
# hIT_rdm_coef = hIT_rdm[np.tril_indices(hIT_rdm.shape[0], k=-1)]
# mean_hIT_rdm_coef = np.mean(hIT_rdm[np.tril_indices(hIT_rdm.shape[0], k=-1)], axis=1)

# r_df = pd.DataFrame(columns=['scale','r'], index=np.arange(len(fm_rdv)))
# for i, (rescale, rdv) in enumerate(fm_rdv.items()):
#     r, _ = pearsonr(rdv, mean_hIT_rdm_coef)
#     print('{}: {:.3f}'.format(rescale, r))
#     r_df['r'][i] = r
#     r_df['scale'][i] = rescale

ratio_time_corr = compute_corr(ratio_fm_rdv, ratio_resize_scale, mean_hIT_rdm)
fixed_time_corr = compute_corr(fixed_fm_rdv, resize_scale, mean_hIT_rdm)


# sub level analysis
# time_corr = {'ratio':[], 'fixed':[]}
# rdm_shape = (mean_hIT_rdm.shape[0], mean_hIT_rdm.shape[1], mean_hIT_rdm.shape[2], 1)
# for i in np.arange(mean_hIT_rdm.shape[-1]):
#     ratio_time_corr = compute_corr(ratio_fm_rdv, ratio_resize_scale, mean_hIT_rdm[:,:,:,i].reshape(rdm_shape))
#     fixed_time_corr = compute_corr(fixed_fm_rdv, resize_scale, mean_hIT_rdm[:,:,:,i].reshape(rdm_shape))
#     time_corr['ratio'].append(ratio_time_corr)
#     time_corr['fixed'].append(fixed_time_corr)
#     print(f'sub{i+1} analysis completed')
    
ori_corr = ratio_time_corr['rescale=(1, 1)']

eTime = time.time()
time_cost = eTime - sTime
print('spend {:.0f}h {:.0f}min {:.2f}sec'.format(time_cost//3600, (time_cost%3600)//60, time_cost%60))


# convert to 3D DataFrame
# coln = ['reduce_W_ratio', 'reduce_H_ratio']
# coln.extend([f'layer{i}' for i in layer_list])
# ratio_corr_df = pd.DataFrame(columns=coln, index=np.arange(len(ratio_time_corr.keys())))

# for i, (resize, col) in enumerate(ratio_time_corr.iteritems()):
    
#     for layer_idx in [f'layer{i}' for i in layer_list]:
#         ratio_corr_df[layer_idx][i] = col[layer_idx]
        
#     ratio_corr_df['reduce_W_ratio'][i] = int(re.findall(r"[(](.*?),", resize)[0])
#     ratio_corr_df['reduce_H_ratio'][i] = int(re.findall(r"[ ](.*?)[)]", resize)[0])
 
    
# coln = ['reduce_W_fixed', 'reduce_H_fixed']
# coln.extend([f'layer{i}' for i in layer_list])
# fixed_corr_df = pd.DataFrame(columns=coln, index=np.arange(len(fixed_time_corr.keys())))

# for i, (resize, col) in enumerate(fixed_time_corr.iteritems()):
    
#     for layer_idx in [f'layer{i}' for i in layer_list]:
#         fixed_corr_df[layer_idx][i] = col[layer_idx]
        
#     fixed_corr_df['reduce_W_fixed'][i] = int(re.findall(r"[(](.*?),", resize)[0])
#     fixed_corr_df['reduce_H_fixed'][i] = int(re.findall(r"[ ](.*?)[)]", resize)[0])
    


#%% evaluation and visualization

# print corr that is higher than un-downsampled feature maps
# ori_corr = ratio_time_corr['rescale=(1, 1)']
df_type=['ratio', 'fixed']
ori_corr = ratio_time_corr['rescale=(1, 1)']

eval_score = {}
eval_score['ratio'] = compute_eval_score(ratio_time_corr, ori_corr)
np.save('vgg16_bn_dataset92_mean_meg_eval_scores.npy', eval_score)

for i, time_corr in enumerate([ratio_time_corr, fixed_time_corr]):
    
    eval_score[df_type[i]] = compute_eval_score(time_corr, ori_corr)
    
plot_eval_score(eval_score, ori_corr)


# sub level analysis
# for sub_idx in np.arange(15,16):
    
#     ratio_time_corr, fixed_time_corr = time_corr['ratio'][sub_idx], time_corr['fixed'][sub_idx]
#     ori_corr = ratio_time_corr['rescale=(1, 1)']
    
#     eval_score = {}
    
#     for i, sub_time_corr in enumerate([ratio_time_corr, fixed_time_corr]):
        
#         eval_score[df_type[i]] = compute_eval_score(sub_time_corr, ori_corr)
        
        
#     plot_eval_score(eval_score, ori_corr)
            

        

# # 3D plot
# for i, layer_idx in enumerate([f'layer{i}' for i in layer_list]):
#     scale = np.array(fixed_corr_df.iloc[:,0:2])
#     max_corr = np.array([])
#     for corr_list in fixed_corr_df.iloc[:,2+i]:
#         max_corr = np.append(max_corr, np.max(corr_list))
    
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     fig.add_axes(ax)
#     ax.plot_trisurf(list(scale[:,0]), list(scale[:,1]), max_corr)
#     ax.set_xlabel(fixed_corr_df.keys()[0])
#     ax.set_ylabel(fixed_corr_df.keys()[1])
#     ax.set_zlabel('max r')
#     ax.set_title(layer_idx)
#     fig.savefig(os.path.join(os.getcwd(), 'figs', f'max_corr_{layer_idx}_fixed.png'))

# for i, layer_idx in enumerate([f'layer{i}' for i in layer_list]):
#     scale = np.array(ratio_corr_df.iloc[:,0:2])
#     max_corr = np.array([])
#     for corr_list in ratio_corr_df.iloc[:,2+i]:
#         max_corr = np.append(max_corr, np.max(corr_list))
    
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     fig.add_axes(ax)
#     ax.plot_trisurf(list(scale[:,0]), list(scale[:,1]), max_corr)
#     ax.set_xlabel(ratio_corr_df.keys()[0])
#     ax.set_ylabel(ratio_corr_df.keys()[1])
#     ax.set_zlabel('max r')
#     ax.set_title(layer_idx)
#     fig.savefig(os.path.join(os.getcwd(), 'figs', f'max_corr_{layer_idx}_ratio.png'))


# plot correlation changes by time 
plot_time_corr(ratio_time_corr, ori_corr, 'combined')
# plt.savefig(os.path.join(os.getcwd(), 'figs', 'ratio_time_corr.png'))
plot_time_corr(fixed_time_corr, ori_corr, 'combined')
# plt.savefig(os.path.join(os.getcwd(), 'figs', 'fixed_time_corr.png'))
                
#%% compute ori corr

sTime = time.time()

save_pth = '/nfs/s2/userhome/daiyuxuan/workingdir/MPI'
files_dict = {'features':{'92': './dataset/92_object_imageset',
                             '84_1': './dataset/84_images/group1',
                             '84_2': './dataset/84_images/group2',
                             '118': './dataset/118_object_imageset'},
                 'meg_rdm':{'92': h5py.File('./dataset/92_object_imageset/MEG_decoding_RDMs.mat', 'r'),
                            '84_1': loadmat('./dataset/84_images/meg_rdm.mat')['RDM84_MEG_group1'],
                            '84_2': loadmat('./dataset/84_images/meg_rdm.mat')['RDM84_MEG_group2'],
                            '118': loadmat('./dataset/118_object_imageset/MEG_RDMs/MEG_decoding_RDMs.mat')['MEG_decoding_RDMs'][:,0,:,:,:].T}
                }

ratio_resize_scale = [(1,1)]
ori_corr_dict = {}
    
for dataset_idx in ['84_1', '84_2', '92', '118']:
    
    for model_name in ['vgg16', 'vgg16_bn']:
        
        if model_name == 'vgg16':
            layer_list = [2, 7, 14, 21, 28]
        else:
            layer_list = [3, 10, 17, 24, 27, 37]
        
        features_ori = extract_features(files_dict['features'][dataset_idx], model_name, layer_list)

        ratio_fm_df = compute_FM(features_ori, ratio_resize_scale, scale_type='ratio')
            
        ratio_fm_rdv = compute_rdv(ratio_fm_df, ratio_resize_scale)
        
        meg_rdm = files_dict['meg_rdm'][dataset_idx]
        
        if dataset_idx == '92':
            
            sess1_hIT_rdm = meg_rdm['MEG_decoding_RDMs'][:,:,:,0,:]
            sess2_hIT_rdm = meg_rdm['MEG_decoding_RDMs'][:,:,:,1,:]
            mean_hIT_rdm = np.mean(meg_rdm['MEG_decoding_RDMs'], axis=3)
            ori_corr_dict[f'dataset{dataset_idx}_{model_name}'] = {}
            
            sess1_corr = compute_corr(ratio_fm_rdv, ratio_resize_scale, sess1_hIT_rdm)
            sess2_corr = compute_corr(ratio_fm_rdv, ratio_resize_scale, sess2_hIT_rdm)
            mean_corr = compute_corr(ratio_fm_rdv, ratio_resize_scale, mean_hIT_rdm)
            
            ori_corr_dict[f'dataset{dataset_idx}_{model_name}']['sess1'] = sess1_corr['rescale=(1, 1)']
            ori_corr_dict[f'dataset{dataset_idx}_{model_name}']['sess2'] = sess2_corr['rescale=(1, 1)']
            ori_corr_dict[f'dataset{dataset_idx}_{model_name}']['mean'] = mean_corr['rescale=(1, 1)']
            
        else:
            ratio_time_corr = compute_corr(ratio_fm_rdv, ratio_resize_scale, meg_rdm)
            ori_corr_dict[f'dataset_{dataset_idx}_{model_name}'] = ratio_time_corr['rescale=(1, 1)']

np.save(os.path.join(save_pth, 'ori_corr_dict.npy'), ori_corr_dict)        
    
eTime = time.time()
time_cost = eTime - sTime
print('spend {:.0f}h {:.0f}min {:.2f}sec'.format(time_cost//3600, (time_cost%3600)//60, time_cost%60))


#%%

def select_high_scale(ratio_score, layer_list, high_pass=5):
    
    sig_scales = {}
    
    for layer_idx in layer_list:
    
        layer_ratio_score = ratio_score[f'layer{layer_idx}']
        sig_scales[f'layer{layer_idx}'] = []
        
        for row_index, row in layer_ratio_score.iterrows():
            
            if row['score_diff'] > high_pass:
                sig_scales[f'layer{layer_idx}'].append(row['resize'])
    
    return sig_scales
 
def reorg_scales(vgg_sig_scales):
    
    vgg_scale = vgg_sig_scales[0].copy()
    for i, scales in enumerate(vgg_sig_scales):
        
        if i == 0:
            for keys, values in scales.items():
                if len(values):
                    vgg_scale[keys] = [(int(j[-5]), int(j[-2])) for j in values]
                else:
                    vgg_scale[keys] = []
        
        else:
            for keys, values in scales.items():
                if len(values):
                    vgg_scale[keys].extend([(int(j[-5]), int(j[-2])) for j in values])
            
    for keys in vgg_scale.keys():
        vgg_scale[keys].sort()
        vgg_scale[keys] = np.array(vgg_scale[keys])
        
    return vgg_scale
    

# if model_name == 'vgg16':
# layer_list = [2, 7, 14, 21, 28]
# else:
# layer_list = [3, 10, 17, 24, 27, 37]       

score_list = os.listdir(os.path.join(os.getcwd(), 'eval_scores'))

vgg_sig_scales = []
vggbn_sig_scales = []

for score_fname in score_list:
    
    eval_score = np.load(os.path.join(os.getcwd(), 'eval_scores', score_fname), 
                         allow_pickle=True).all()
    
    if 'vgg16_bn' in score_fname:
        layer_list = [3, 10, 17, 24, 27, 37]
        sig_scales = select_high_scale(eval_score['ratio'], layer_list, high_pass=10)
        vggbn_sig_scales.append(sig_scales)
        
    else:
        layer_list = [2, 7, 14, 21, 28]
        sig_scales = select_high_scale(eval_score['ratio'], layer_list, high_pass=10)
        vgg_sig_scales.append(sig_scales)
        

vgg_scale = reorg_scales(vgg_sig_scales)
vggbn_scale = reorg_scales(vggbn_sig_scales)

for layer_idx, val in vgg_scale.items():
    print('vgg {} best resize scale is {}'.format(layer_idx, np.mean(val, axis=0)))

for layer_idx, val in vggbn_scale.items():
    print('vggbn {} best resize scale is {}'.format(layer_idx, np.mean(val, axis=0)))
    
    
#%%

def select_sig_scale(ratio_score, ori_data, layer_list, sig_level=0.05):
    
    sig_scales = {}
    
    for layer_idx in layer_list:
    
        layer_ratio_score = ratio_score[f'layer{layer_idx}']
        layer_ori_corr = ori_data[f'layer{layer_idx}']
        sig_scales[f'layer{layer_idx}'] = []
        
        for row_index, row in layer_ratio_score.iterrows():
            
            t_stat, p = ttest_rel(row['corr'], layer_ori_corr)
            
            if p < sig_level and row['score_diff'] > 5:
                
                sig_scales[f'layer{layer_idx}'].append(row['resize'])
    
    return sig_scales

score_ori = [('dataset84_meg_group2_eval_scores.npy', ori_corr_dict['dataset84_2_vgg16']),
             ('vgg16_bn_dataset84_group2_meg_eval_scores.npy', ori_corr_dict['dataset84_2_vgg16_bn']),
             ('vgg16_bn_dataset84_group1_meg_eval_scores.npy', ori_corr_dict['dataset84_1_vgg16_bn']),
             ('dataset118_meg_eval_scores.npy', ori_corr_dict['dataset118_vgg16']),
             ('vgg16_bn_dataset118_meg_eval_scores.npy', ori_corr_dict['dataset118_vgg16_bn']),
             ('dataset92_meg_sees2_eval_scores.npy', ori_corr_dict['dataset92_vgg16']['sess2']),
             ('dataset92_meg_mean_eval_scores.npy', ori_corr_dict['dataset92_vgg16']['mean']), 
             ('dataset92_meg_sees1_eval_scores.npy', ori_corr_dict['dataset92_vgg16']['sess1']),
             ('vgg16_bn_dataset92_mean_meg_eval_scores.npy', ori_corr_dict['dataset92_vgg16_bn']['mean']),
             ('vgg16_bn_dataset92_sess1_meg_eval_scores.npy', ori_corr_dict['dataset92_vgg16_bn']['sess1']),
             ('vgg16_bn_dataset92_sess2_meg_eval_scores.npy', ori_corr_dict['dataset92_vgg16_bn']['sess2'])]


vgg_sig_scales = []
vggbn_sig_scales = []
for score_fname, ori_data in score_ori:
    
    eval_score = np.load(os.path.join(os.getcwd(), 'eval_scores', score_fname), 
                         allow_pickle=True).all()
    
    if 'vgg16_bn' in score_fname:
        layer_list = [3, 10, 17, 24, 27, 37]
        sig_scales = select_sig_scale(eval_score['ratio'], ori_data, layer_list, sig_level=0.01)
        vggbn_sig_scales.append(sig_scales)
        
    else:
        layer_list = [2, 7, 14, 21, 28]
        sig_scales = select_sig_scale(eval_score['ratio'], ori_data, layer_list, sig_level=0.01)
        vgg_sig_scales.append(sig_scales)
        
vgg_scale = reorg_scales(vgg_sig_scales)
vggbn_scale = reorg_scales(vggbn_sig_scales)

for layer_idx, val in vgg_scale.items():
    print('vgg {} best resize scale is {}'.format(layer_idx, np.mean(val, axis=0)))

for layer_idx, val in vggbn_scale.items():
    print('vggbn {} best resize scale is {}'.format(layer_idx, np.mean(val, axis=0)))
    