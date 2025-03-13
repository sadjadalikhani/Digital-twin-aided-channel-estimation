#%% SITE SEGMENTATION
import numpy as np
import matplotlib.pyplot as plt
from utils import k_means, k_med, subspace_estimation, todB, chs_gen, subspace_estimation_drl, generate_dft_codebook, plot_smooth_cdf, plot_perf_vs_pilots
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")
#%% DT AND RW CHANNEL GENERATION
scenarios = ['indianapolis_4mShifted_28GHz', 'indianapolis_original_28GHz'] # 'indianapolis_2mShifted_28GHz'
n_beams = 128 
fov = 180
n_path = [1, 25] 

M_x = 1
M_y = n_beams // M_x
codebook = generate_dft_codebook(M_x, M_y) 

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
#%% FIG. 2: PERF VS PILOTS PLOT (NO CALIBRATION)
trials = 200
datasets = [
    "Real-World",  
    "Digital Twin",  
    "Random DFT-based Pilots"
]
dft_based = True  
n_pilots = np.array([1, 13, 26, 38, 51, 64, 77, 90, 102, 115, 128])
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
        
        if (dataset_idx == 0 and subspace_coeff == 0) or dataset_type in ["Random DFT-based Pilots"]:
            
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
    plot_perf_vs_pilots(datasets, ss_nmse, n_pilots, n_beams, trial, loss_func)
#%% FIG. 3: CDF (WITH CALIBRATION)
trials = 200
datasets = [
    "Real-World",  
    "Digital Twin",  
    "RL-Calibrated Digital Twin",  
    "Random DFT-based Pilots",  
    "RL-Calibrated Random Pilots"
]
dft_based = True  
n_pilots = [26]
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
        
        if (dataset_idx == 0 and subspace_coeff == 0) or dataset_type in ["Random DFT-based Pilots"]:
            
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
    datasets = [
        "Real-world",  
        "Digital twin",  
        "RL-calibrated digital twin",  
        "Random DFT-based pilots",  
        "RL-calibrated random pilots"
    ]
    plot_smooth_cdf(datasets, ss_nmse, trial=trial, loss_func=loss_func, n=13, r=0.2)

