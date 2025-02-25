# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:50:54 2024

@author: salikha4
"""
#%% SITE SEGMENTATION
import numpy as np
import matplotlib.pyplot as plt
from utils import k_means, k_med, subspace_estimation, todB, chs_gen, subspace_estimation_drl
import torch
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

#%% DT AND RW CHANNEL GENERATION
scenarios = ['indianapolis_original_28GHz', 'indianapolis_2mShifted_28GHz']  
n_beams = 128 
fov = 180
n_path = [5, 1] 

M_x = 16
M_y = n_beams // M_x
dft_N = np.fft.fft(np.eye(M_x)) / np.sqrt(M_x)
dft_M = np.fft.fft(np.eye(M_y)) / np.sqrt(M_y)
codebook = np.kron(dft_M, dft_N)
codebook = torch.tensor(codebook).to(torch.complex64)

dataset_dt, dataset_rw, pos, los_status, best_beam, enabled_idxs, bs_pos = chs_gen(
    scenarios,
    n_beams, 
    fov, 
    n_path,
    codebook)

#%% settings
n_users = len(dataset_dt)
pos_coeff = 1
los_coeff_kmeans = 0
beam_coeff_kmeans = 0 
umap_coeff = 0
subspace_coeff = 0

#%% FIG. 2: CDF (WITH CALIBRATION)
trials = 200
datasets = [
    "Real-World",  
    "Digital Twin",  
    "RL-Calibrated Digital Twin",  
    "Random DFT-based Pilots",  
    "RL-Calibrated Random Pilots"
]
dft_based = True  
n_pilots = [19]
snr_db = 10 
loss_func = ["nmse", "cosine", "throughput"][1]
ss_nmse = np.zeros((len(datasets), len(n_pilots), trials))

for trial in range(trials):
    
    for dataset_idx, dataset_type in enumerate(datasets):
        
        print(f"\n\ntrial: {trial}\ndataset type: {dataset_type}")
        
        n_areas = 12
        n_kmeans_clusters = 80 
        
        if dataset_type in ["Digital Twin", "RL-Calibrated Digital Twin"]:
            imperfect_dataset = dataset_dt
        elif dataset_type == "Real-World":
            imperfect_dataset = dataset_rw
        elif dataset_type in ["Random DFT-based Pilots", "RL-Calibrated Random Pilots"]:
            imperfect_dataset = dataset_rw
            n_areas = 1
            n_kmeans_clusters = 1
        
        dt_subspaces, rw_subspaces, kmeans_centroids, kmeans_labels = k_means(
            enabled_idxs, 
            imperfect_dataset, 
            dataset_rw,
            pos[:,:3], 
            los_status[0 if dataset_idx in [0, 3, 4] else 1 if dataset_idx in [1, 2] else dataset_idx] if len(los_status) > 1 else los_status,
            best_beam[0 if dataset_idx in [0, 3, 4] else 1 if dataset_idx in [1, 2] else dataset_idx] if len(best_beam) > 1 else best_beam,
            bs_pos[0], 
            pos_coeff,  
            los_coeff_kmeans, 
            beam_coeff_kmeans,  
            percent=.95,  
            n_kmeans_clusters=n_kmeans_clusters, 
            k_predefined2=None,
            seed=trial
        )
        
        areas, area_lens = k_med(
            dt_subspaces, 
            pos_coeff, 
            subspace_coeff, 
            kmeans_centroids, 
            n_areas, 
            kmeans_labels,
            pos[:,:3],
            enabled_idxs,
            bs_pos[0],
            seed=trial
        )
    
        if dataset_type in ["Real-World", "Random DFT-based Pilots", "Digital Twin"]:
            
            avg_nmse_ss = subspace_estimation(
                imperfect_dataset, 
                dataset_rw, 
                areas, 
                area_lens, 
                codebook,
                n_pilots,
                dataset_type,
                snr_db=snr_db,
                loss_func=loss_func,
                dft_based=dft_based,
                seed=trial
            )
            
        elif dataset_type in ["RL-Calibrated Digital Twin", "RL-Calibrated Random Pilots"]:
            
            avg_nmse_ss = subspace_estimation_drl(
                imperfect_dataset, 
                dataset_rw, 
                areas, 
                area_lens, 
                codebook, 
                dataset_type,
                n_pilots=n_pilots[0], 
                n_episodes=300, 
                snr_db=snr_db, 
                loss_func=loss_func, 
                seed=trial
            )
        
        ss_nmse[dataset_idx, :, trial] = todB(avg_nmse_ss) if loss_func == "nmse" else avg_nmse_ss
    # CDF PLOT
    colormap = cm.get_cmap('magma', int(3 * 1.5))
    colors = [colormap(i / (3 - 1)) for i in range(len(datasets))]

    # Define line styles
    line_styles = ['-', '-', '--', '-', '--']  # Third is dashed version of second, fifth is dashed version of fourth

    plt.figure(dpi=1000)

    for i in range(ss_nmse.shape[0]):
        sorted_vals = np.sort(ss_nmse[i][0][:trial+1])
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        plt.step(sorted_vals, cdf, where='post', 
                 color=colors[i if i in [0, 1, 3] else i-1],  # Apply color from Spectral for 1st, 2nd, 4th; 3rd & 5th inherit from 2nd & 4th
                 linestyle=line_styles[i],  # Apply dashed styles for 3rd and 5th
                 label=f"{datasets[i]}", linewidth=2.5)

    plt.xlabel(
        "NMSE (dB)" if loss_func == "nmse" else 
        "Cosine Similarity" if loss_func == "cosine" else 
        "Throughput", 
        fontsize=12
    )
    plt.ylabel('Empirical CDF', fontsize=12)
    plt.legend(fontsize=10, loc='best', frameon=True, facecolor='white', edgecolor='black')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
#%% FIG. 3: PERF VS PILOTS PLOT (NO CALIBRATION)
trials = 200
datasets = [
    "Real-World",  
    "Digital Twin",  
    "Random DFT-based Pilots"
]
dft_based = True  
n_pilots = np.array([1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128])
snr_db = 10 
loss_func = ["nmse", "cosine", "throughput"][1]
ss_nmse = np.zeros((len(datasets), len(n_pilots), trials))

for trial in range(trials):
    
    for dataset_idx, dataset_type in enumerate(datasets):
        
        print(f"\n\ntrial: {trial}\ndataset type: {dataset_type}")
        
        n_areas = 12 
        n_kmeans_clusters = 80 
        
        if dataset_type in ["Digital Twin"]:
            imperfect_dataset = dataset_dt
        elif dataset_type == "Real-World":
            imperfect_dataset = dataset_rw
        elif dataset_type == "Random DFT-based Pilots":
            imperfect_dataset = dataset_rw
            n_areas = 1
            n_kmeans_clusters = 1
        
        dt_subspaces, rw_subspaces, kmeans_centroids, kmeans_labels = k_means(
            enabled_idxs, 
            imperfect_dataset, 
            dataset_rw,
            pos[:,:3], 
            los_status[0 if dataset_idx in [0, 2] else 1 if dataset_idx in [1] else dataset_idx] if len(los_status) > 1 else los_status,
            best_beam[0 if dataset_idx in [0, 2] else 1 if dataset_idx in [1] else dataset_idx] if len(best_beam) > 1 else best_beam,
            bs_pos[0], 
            pos_coeff,  
            los_coeff_kmeans, 
            beam_coeff_kmeans,  
            percent=.95,  
            n_kmeans_clusters=n_kmeans_clusters, 
            k_predefined2=None,
            seed=trial
        )
        
        areas, area_lens = k_med(
            dt_subspaces, 
            pos_coeff, 
            subspace_coeff, 
            kmeans_centroids, 
            n_areas, 
            kmeans_labels,
            pos[:,:3],
            enabled_idxs,
            bs_pos[0],
            seed=trial
        )
        
        
        avg_nmse_ss = subspace_estimation(
            imperfect_dataset, 
            dataset_rw, 
            areas, 
            area_lens, 
            codebook,
            n_pilots,
            dataset_type,
            snr_db=snr_db,
            loss_func=loss_func,
            dft_based=dft_based,
            seed=trial
        )
        
        ss_nmse[dataset_idx, :, trial] = todB(avg_nmse_ss).squeeze() if loss_func == "nmse" else avg_nmse_ss.squeeze()

    # FIGURE    
    colormap = cm.get_cmap('Spectral', int(len(datasets) * 1.5))
    colors = [colormap(i / (len(datasets) - 1)) for i in range(len(datasets))]
    line_styles = ['-', '-','-','-.', '--', ':']
    markers = ['D', 's', 'o', '^', 'v', 'p', 'h', '*']  
    plt.figure(dpi=1000)
    
    for i in range(len(datasets)):
    
        x_values = n_pilots / n_beams * 100 
        y_values = np.mean(ss_nmse[i][:, :trial+1], axis=-1) 
    
        linestyle = line_styles[i % len(line_styles)]  
        marker = markers[i % len(markers)]  
    
        plt.plot(x_values, y_values, label=f"{datasets[i]}", color=colors[i], linewidth=2, 
                 linestyle=linestyle, marker=marker, markerfacecolor='white', 
                 markeredgewidth=2, markeredgecolor=colors[i])
    
    plt.xlabel("Number of Pilots (%)", fontsize=12)
    plt.ylabel("NMSE (dB)" if loss_func == "nmse" else "Cosine Similarity" if loss_func == "cosine" else "Throughput", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=":", alpha=0.3)  # Minor grid with dotted lines
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


