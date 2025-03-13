# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:13:29 2024

This script generates preprocessed data from wireless communication scenarios, 
including token generation, patch creation, and data sampling for machine learning models.

@author: salikha4
"""

import numpy as np
from tqdm import tqdm
import DeepMIMOv3

#%%
def deepmimo_data_cleaning(deepmimo_data):
    idxs = np.where(deepmimo_data['user']['LoS'] != -1)[0]
    cleaned_deepmimo_data = deepmimo_data['user']['channel'][idxs]
    return np.array(cleaned_deepmimo_data) * 1e6

#%% Data Generation for Scenario Areas
def DeepMIMO_data_gen(scenario, n_path=10):

    import DeepMIMOv3
    
    parameters, row_column_users, n_ant_bs, n_ant_ue, n_subcarriers = get_parameters(scenario, n_path)
    
    deepMIMO_dataset = DeepMIMOv3.generate_data(parameters)
    uniform_idxs = uniform_sampling(deepMIMO_dataset, [1, 1], len(parameters['user_rows']), 
                                    users_per_row=row_column_users[scenario]['n_per_row'])
    data = select_by_idx(deepMIMO_dataset, uniform_idxs)[0]
        
    return data

#%%%
def get_parameters(scenario, n_path):
    
    n_ant_bs = 16 #16 #32
    n_ant_bs_vert = 8 #8
    n_ant_ue = 1
    n_subcarriers = 1 #32
    scs = 30e3
        
    row_column_users = {
    'indianapolis_original_28GHz': {
        'n_rows': 80,
        'n_per_row': 79
    },
    'indianapolis_2mShifted_28GHz': {
        'n_rows': 80,
        'n_per_row': 79
    },
    'indianapolis_4mShifted_28GHz': {
        'n_rows': 80,
        'n_per_row': 79
    },
    'iceland_rw': {
        'n_rows': 74,
        'n_per_row': 83
    },
    'iceland_dt': {
        'n_rows': 74,
        'n_per_row': 83
    },
    'barcelona_dt': {
        'n_rows': 64,
        'n_per_row': 65
    },
    'barcelona_rw': {
        'n_rows': 64,
        'n_per_row': 65
    }}


    parameters = DeepMIMOv3.default_params()
    parameters['dataset_folder'] = './scenarios'
    parameters['scenario'] = scenario
    parameters['active_BS'] = np.array([3])  # np.array([3]) 
    parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'])
    parameters['bs_antenna']['shape'] = np.array([n_ant_bs, n_ant_bs_vert]) # Horizontal, Vertical 
    parameters['bs_antenna']['rotation'] = np.array([0,0,-135]) # (x,y,z)
    parameters['bs_antenna']['FoV'] = np.array([180, 180])
    parameters['ue_antenna']['shape'] = np.array([n_ant_ue, 1])
    parameters['enable_BS2BS'] = False
    parameters['OFDM']['subcarriers'] = n_subcarriers
    parameters['OFDM']['selected_subcarriers'] = np.arange(n_subcarriers)
    parameters['OFDM']['bandwidth'] = scs * n_subcarriers / 1e9
    parameters['num_paths'] = n_path
    
    return parameters, row_column_users, n_ant_bs, n_ant_ue, n_subcarriers

#%% Sampling and Data Selection
def uniform_sampling(dataset, sampling_div, n_rows, users_per_row):

    cols = np.arange(users_per_row, step=sampling_div[0])
    rows = np.arange(n_rows, step=sampling_div[1])
    uniform_idxs = np.array([j + i * users_per_row for i in rows for j in cols])
    
    return uniform_idxs

def select_by_idx(dataset, idxs):
  
    dataset_t = []  # Trimmed dataset
    for bs_idx in range(len(dataset)):
        dataset_t.append({})
        for key in dataset[bs_idx].keys():
            dataset_t[bs_idx]['location'] = dataset[bs_idx]['location']
            dataset_t[bs_idx]['user'] = {k: dataset[bs_idx]['user'][k][idxs] for k in dataset[bs_idx]['user']}
    
    return dataset_t

#%% Label Generation
def label_gen(task, data, scenario, F1, n_beams=64, n_path=10, fov=180, noise=None, manual_filter=None):
    import utils as dt
    if manual_filter is None:
        idxs = np.where(data['user']['LoS'] != -1)[0]
    else:
        idxs = manual_filter
           
    F1 = np.array(F1)
    
    if task == 'LoS/NLoS Classification':
        F1 = 0
        label = data['user']['LoS'][idxs]
        
        var_names = ['LoS']
        for var_name in var_names:
            if var_name == 'LoS':
                losChs = data['user'][var_name][idxs]
                # losChs = np.where(losChs == -1, np.nan, losChs)
                plot_coverage(data['user']['location'][idxs], losChs, tx_pos=data['location'], 
                                 title=var_name, cbar_title=var_name)
                
    elif task == 'Beam Prediction':
        parameters, row_column_users, n_ant_bs, n_ant_ue, n_subcarriers = get_parameters(scenario, n_path)
        n_users = len(data['user']['channel'])
        n_subbands = 1

        full_dbm = np.zeros((n_beams, n_subbands, n_users), dtype=float)
        for ue_idx in tqdm(range(n_users), desc='Computing the channel for each user'):
            if data['user']['LoS'][ue_idx] == -1:
                full_dbm[:,:,ue_idx] = np.nan
            else:
                if noise is not None:
                    chs = F1 @ (data['user']['channel'][ue_idx] + np.array(noise[ue_idx]))
                else:
                    chs = F1 @ data['user']['channel'][ue_idx]
                full_linear = np.abs(np.mean(chs.squeeze().reshape((n_beams, n_subbands, -1)), axis=-1))
                full_dbm[:,:,ue_idx] = np.around(20*np.log10(full_linear) + 30, 1)

        best_beams = np.argmax(np.mean(full_dbm,axis=1), axis=0)
        best_beams = best_beams.astype(float)
        best_beams[np.isnan(full_dbm[0,0,:])] = np.nan
        # max_bf_pwr = np.max(np.mean(full_dbm,axis=1), axis=0) 
    
        label = best_beams[idxs]
        
        plot_coverage(data['user']['location'], best_beams, tx_pos=data['location'], 
                      tx_ori=parameters['bs_antenna']['rotation']*np.pi/180, 
                      title= 'Best Beams', cbar_title='Best beam index')
        
    # return label.astype(int)
    label = label.astype(float)  # Keep as float
    label[np.isnan(label)] = -1  # Ensure NaN values remain unchanged
    return label

def steering_vec(array, phi=0, theta=0, kd=np.pi):
    idxs = DeepMIMOv3.ant_indices(array)
    resp = DeepMIMOv3.array_response(idxs, phi, theta+np.pi/2, kd)
    return resp / np.linalg.norm(resp)

def create_labels(task, scenario_names, n_beams=64):
    labels = []
    for scenario_name in scenario_names:
        data = DeepMIMO_data_gen(scenario_name)
        labels.extend(label_gen(task, data, scenario_name, n_beams=n_beams))
    return labels
#%%
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_coverage(rxs, cov_map, dpi=500, figsize=(6,4), cbar_title=None, title=False,
                  scat_sz=20, tx_pos=None, tx_ori=None, legend=False, lims=None,
                  proj_3D=False, equal_aspect=False, tight=True, cmap='tab20'):
    
    # Determine the number of unique labels in cov_map and create a discrete colormap
    num_labels = len(np.unique(cov_map))
    cmap = ListedColormap(plt.get_cmap(cmap).colors[:num_labels]) if num_labels <= 20 else plt.get_cmap(cmap)
    
    plt_params = {'cmap': cmap}
    if lims:
        plt_params['vmin'], plt_params['vmax'] = lims[0], lims[1]
    
    n = 3 if proj_3D else 2  # n coordinates to consider 2 = xy | 3 = xyz
    
    xyz = {'x': rxs[:, 0], 'y': rxs[:, 1]}
    if proj_3D:
        xyz['zs'] = rxs[:, 2]
        
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize,
                           subplot_kw={'projection': '3d'} if proj_3D else {})
    
    im = ax.scatter(**xyz, c=cov_map, s=scat_sz, marker='s', **plt_params)

    cbar = plt.colorbar(im, label='' if not cbar_title else cbar_title)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    # TX position
    if tx_pos is not None:
        ax.scatter(*tx_pos[:n], marker='P', c='r', label='TX')
    
    # TX orientation
    if tx_ori is not None and tx_pos is not None:  # ori = [azi, el]
        r = 30  # ref size of pointing direction
        tx_lookat = np.copy(tx_pos)
        tx_lookat[:2] += r * np.array([np.cos(tx_ori[2]), np.sin(tx_ori[2])])  # azimuth
        tx_lookat[2] += r * np.sin(tx_ori[1])  # elevation
        
        line_components = [[tx_pos[i], tx_lookat[i]] for i in range(n)]
        line = {key: val for key, val in zip(['xs', 'ys', 'zs'], line_components)}
        if n == 2:
            ax.plot(line_components[0], line_components[1], c='k', alpha=1, zorder=3)
        else:
            ax.plot(**line, c='k', alpha=1, zorder=3)
    
    if title:
        ax.set_title(title)
    
    if legend:
        plt.legend(loc='upper center', ncols=10, framealpha=0.5)
    
    if tight:
        s = 1
        mins, maxs = np.min(rxs, axis=0) - s, np.max(rxs, axis=0) + s
        if not proj_3D:
            ax.set_xlim([mins[0], maxs[0]])
            ax.set_ylim([mins[1], maxs[1]])
        else:
            ax.set_xlim3d([mins[0], maxs[0]])
            ax.set_ylim3d([mins[1], maxs[1]])
            if tx_pos is None:
                ax.set_zlim3d([mins[2], maxs[2]])
            else:
                ax.set_zlim3d([np.min([mins[2], tx_pos[2]]), np.max([maxs[2], tx_pos[2]])])
    
    if equal_aspect and not proj_3D:
        ax.set_aspect('equal', 'box')
    
    return fig, ax, cbar


class Area():
    def __init__(self, idxs=None, name='', center=''):
        # idxs inside the area
        self.idxs = idxs
        self.name = name
        self.center = center
    
    def __repr__(self):
        s =  f'name = {self.name}\n'
        s += f'center = {self.center}\n'
        s += f'Number of idxs = {len(self.idxs)}\n'
        s += f'idxs = {self.idxs}'
        return s

def plot_areas(areas, all_pos, bs_loc, s=20, show=True, details=True):
    n_areas = len(areas)
    colors = get_colors(n_areas)
    
    f = plt.figure(dpi=200)
    ax = f.add_subplot(111)
    for k, col in zip(range(n_areas), colors):  
        cluster_center = areas[k].center
        plt.scatter(all_pos[areas[k].idxs, 0], all_pos[areas[k].idxs, 1], color=col, s=s, edgecolor=col, linewidths=1)
        if details:
            plt.plot(cluster_center[0], cluster_center[1], "o", 
                      markerfacecolor=col, markeredgecolor="k", markersize=6)
            plt.text(cluster_center[0]+5, cluster_center[1], f'{k}', fontdict={'fontsize':10},
                      bbox=dict(facecolor='white', alpha=0.3))
        plt.plot(bs_loc[0], bs_loc[1], "x", 
                  markerfacecolor=col, markeredgecolor="k", markersize=10, markeredgewidth=2)
        
    plt.title(f'site divided into {n_areas} zones')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.ylim([np.min(all_pos[:,1]), np.max(all_pos[:,1])])
    plt.xlim([np.min(all_pos[:,0]), np.max(all_pos[:,0])])
    if show:
        plt.show()
    return f, ax

def get_colors(n):
    if n <= 10:
        cmap = plt.get_cmap('tab10')  # For small n, use 'tab10'
        return [cmap(i) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap('tab20')  # For medium n, use 'tab20'
        return [cmap(i) for i in range(n)]
    else:
        # For large n, use a continuous colormap (e.g., 'viridis')
        cmap = plt.get_cmap('viridis')  
        return [cmap(i / n) for i in range(n)]  # Normalize index to sample evenly