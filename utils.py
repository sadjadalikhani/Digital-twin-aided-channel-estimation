#%% IMPORTS
from sklearn.cluster import KMeans
import numpy as np
from input_preprocess import DeepMIMO_data_gen, label_gen 
import torch
import matplotlib.pyplot as plt
from kmedoids import KMedoids
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import input_preprocess as preprocess

#%% LOSS FUNCTIONS
def nmse_loss(pred, target):
    nmse_users = torch.norm(target - pred, dim=1)**2 / torch.norm(target, dim=1)**2
    nmse_db = torch.mean(nmse_users)
    return nmse_db

def todB(nmse_db):
    return 10 * torch.log10(torch.tensor(nmse_db).clone().detach())

def cosine_sim(a, b):
    numerator = torch.abs(torch.sum(torch.conj(a) * b)) # real
    denominator = torch.norm(a) * torch.norm(b)
    return numerator / denominator

#%% SUBSPACE CALC
def find_subspace_90_percent(data, percent=.90, k_predefined_2=None):
    
    data = data.clone().detach().type(torch.complex64)
    data_mean = torch.mean(data)
    dt_covariance = (data-data_mean).T.conj() @ (data-data_mean) / (data.size(0))
    
    U, S, Vh = torch.linalg.svd(dt_covariance, full_matrices=False)
    
    total_power = torch.sum(S)
    explained_power = torch.cumsum(S, dim=0) / total_power
    indices = torch.where(explained_power >= percent)[0]
    
    if len(indices) == 0:
        k_90 = len(S) 
    else:
        k_90 = indices[0].item() + 1
        
    if k_predefined_2 != None:
        k_90 = k_predefined_2  
        
    # print(f'{k_90} dimensions')
    U_k = U[:, :k_90]
    
    return U_k, k_90

#%% PROJECTION AND RECONSTRUCTION
def project_onto_subspace(data, U_k):
    data_projected = data @ U_k
    return data_projected

def reconstruct_from_subspace(data_projected, U_k, snr_db=None, seed=42):
    
    if snr_db is not None:
        
        torch.manual_seed(seed) 
        
        signal_power_per_user = torch.mean(torch.abs(data_projected) ** 2, dim=1, keepdim=True)
        snr_linear = 10 ** (snr_db / 10)
        intended_noise_power_per_user = signal_power_per_user / snr_linear
        real_noise = torch.randn_like(data_projected.real)
        imag_noise = torch.randn_like(data_projected.imag)
        raw_complex_noise = real_noise + 1j * imag_noise
        raw_noise_power_per_user = torch.mean(torch.abs(raw_complex_noise) ** 2, dim=1, keepdim=True)
        scaling_factor = torch.sqrt(intended_noise_power_per_user / raw_noise_power_per_user)
        scaled_noise = raw_complex_noise * scaling_factor
        data_projected = data_projected + scaled_noise
        
    else: 
        data_projected = data_projected
        
    data_reconstructed = data_projected @ U_k.conj().T
    
    return data_reconstructed, intended_noise_power_per_user

#%% PCA VISUALIZATION
def visualize_principal_components(data_centered, U, selected_area_idx):
    
    data_projected_pc1 = data_centered @ U[:, 0] 
    data_projected_pc2 = data_centered @ U[:, 1] 
    plt.figure(figsize=(8, 6))
    plt.scatter(data_projected_pc1.numpy(), data_projected_pc2.numpy(), alpha=0.6)
    plt.title(f'Projection onto First Two Principal Components (Area {selected_area_idx})')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.grid(True)
    plt.show()
    explained_var_pc1 = torch.var(data_projected_pc1) / torch.var(data_centered).sum()
    explained_var_pc2 = torch.var(data_projected_pc2) / torch.var(data_centered).sum()
    print(f'PC1 explains {explained_var_pc1.item() * 100:.2f}% of variance')
    print(f'PC2 explains {explained_var_pc2.item() * 100:.2f}% of variance')

#%% DISTANCE MATRIX AND SUBSPACE DISTANCES
def principal_angles(subspace1, subspace2):
    Q1, _ = torch.linalg.qr(subspace1, mode='reduced')
    Q2, _ = torch.linalg.qr(subspace2, mode='reduced')
    M = Q1.conj().T @ Q2
    _, S, _ = torch.linalg.svd(M) 
    S_clamped = torch.clamp(S, -0.999999, 0.999999) 
    principal_angles = torch.acos(S_clamped)
    return principal_angles

def grassmannian_distance(subspace1, subspace2):
    angles = principal_angles(subspace1, subspace2)
    return torch.norm(angles)

def chordal_distance(subspace1, subspace2):
    angles = principal_angles(subspace1, subspace2)
    return torch.norm(torch.sin(angles))

def compute_distance_matrix(subspaces, pos_coeff, subspace_coeff, pos):
    num_users = len(subspaces)
    distance_matrix = np.zeros((num_users, num_users))

    # Compute spatial and subspace distances
    spatial_distances = np.zeros((num_users, num_users))
    subspace_distances = np.zeros((num_users, num_users))

    for i in range(num_users):
        for j in range(i+1, num_users):
            subspace_distances[i, j] = chordal_distance(subspaces[i], subspaces[j]) 
            spatial_distances[i, j] = np.linalg.norm(pos[i] - pos[j])

    # Convert to symmetric matrices
    spatial_distances += spatial_distances.T
    subspace_distances += subspace_distances.T

    # Normalize spatial distances
    spatial_mean, spatial_std = np.mean(spatial_distances), np.std(spatial_distances)
    spatial_distances = (spatial_distances - spatial_mean) / (spatial_std + 1e-8)  # Z-score normalization

    # Normalize subspace distances
    subspace_mean, subspace_std = np.mean(subspace_distances), np.std(subspace_distances)
    subspace_distances = (subspace_distances - subspace_mean) / (subspace_std + 1e-8)

    # Scale to [0,1] range with mean ~ 0.5
    spatial_distances = (spatial_distances - np.min(spatial_distances)) / (np.max(spatial_distances) - np.min(spatial_distances) + 1e-8)
    subspace_distances = (subspace_distances - np.min(subspace_distances)) / (np.max(subspace_distances) - np.min(subspace_distances) + 1e-8)

    # Adjust mean of subspace distances to match that of spatial distances
    subspace_distances *= (np.mean(spatial_distances) / np.mean(subspace_distances + 1e-8))
    

    # Compute final weighted distance matrix
    for i in range(num_users):
        for j in range(i+1, num_users):
            total_distance = subspace_coeff * subspace_distances[i, j] + pos_coeff * spatial_distances[i, j]
            distance_matrix[i, j] = total_distance
            distance_matrix[j, i] = total_distance  # Symmetric matrix

    return distance_matrix


#%% PLOT SUBSPACE
def plot_subspace(U1, U2):
    # S = np.linalg.svd(U1.T @ U2, compute_uv=False)
    # principal_angles = np.rad2deg(np.arccos(S))
    M = U1.conj().T @ U2
    _, sigma, _ = torch.linalg.svd(M)
    sigma_clipped = torch.clamp(sigma, min=-1, max=1)  
    p_angles = torch.acos(sigma_clipped)
    grassmann = torch.norm(p_angles)

    U1 = np.array(U1)
    U2 = np.array(U2)
        
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111, projection='3d', facecolor='none')  # Transparent background
    
    # # Plot original U1 (Real-world Subspace)
    # ax.quiver(0, 0, 0, U1[0, 0], U1[1, 0], U1[2, 0], color='black') # , label='Real-world Subspace'
    # ax.quiver(0, 0, 0, U1[0, 1], U1[1, 1], U1[2, 1], color='black')
    # ax.quiver(0, 0, 0, U1[0, 2], U1[1, 2], U1[2, 2], color='black')
    
    # # Plot target U2 (Digital Twin Subspace)
    # ax.quiver(0, 0, 0, U2[0, 0], U2[1, 0], U2[2, 0], color='red') #, label='Digital Twin Subspace'
    # ax.quiver(0, 0, 0, U2[0, 1], U2[1, 1], U2[2, 1], color='red')
    # ax.quiver(0, 0, 0, U2[0, 2], U2[1, 2], U2[2, 2], color='red')
    
    # # Set labels and title
    # # ax.set_title(f'Grassmann Distance: {grassmann:.3f}', pad=20, fontsize=12, color="black")
    # # ax.legend(fontsize=10, loc='upper left')
    
    # # Remove axis ticks and grid lines for cleaner look
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.grid(False)
    
    # # Remove axis labels
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    # ax.set_zlabel('')
    
    # # Adjust the aspect ratio to ensure a natural 3D look
    # ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    
    # import time
    # filename = f"Desktop\3d_plot_{int(time.time())}.png"
    # plt.savefig(filename, dpi=300, transparent=True)
    
    # # Show the plot
    # plt.show()

        
    return p_angles, grassmann
#%% KMEANS
def k_means(
        enabled_idxs, 
        dataset_dt, 
        dataset_rw,
        pos, 
        los_status, 
        best_beam, 
        bs_pos, 
        pos_coeff,  
        los_coeff_kmeans, 
        beam_coeff_kmeans,  
        percent=.9, 
        n_kmeans_clusters=50, 
        k_predefined2=None,
        seed=42
    ):
    
    features = np.hstack((pos * pos_coeff, los_status * los_coeff_kmeans, best_beam * beam_coeff_kmeans))
    kmeans = KMeans(n_clusters=n_kmeans_clusters, random_state=seed)
    kmeans.fit(features)
    kmeans_labels = kmeans.labels_
    kmeans_centroids = kmeans.cluster_centers_

    kmeans_areas = [preprocess.Area(enabled_idxs[np.where(kmeans.labels_ == i)[0]], 
                      name=i, center=kmeans_centroids[i][:2]/pos_coeff) for i in range(n_kmeans_clusters)]
    preprocess.plot_areas(kmeans_areas, pos, bs_pos, details=None)

    dim=[]
    dt_subspaces = []
    rw_subspaces = []
    
    for k in range(n_kmeans_clusters):
        
        cluster_indices = np.where(kmeans_labels == k)[0]
        stacked_channels = torch.cat([dataset_dt[i].unsqueeze(0) for i in cluster_indices], dim=0)
        
        stacked_channels_rw = torch.cat([dataset_rw[i].unsqueeze(0) for i in cluster_indices], dim=0)
            
        dominant_subspace, k_percent = find_subspace_90_percent(stacked_channels, 
                                                                percent=percent, 
                                                                k_predefined_2=k_predefined2) 
        
        dominant_subspace_rw, k_percent_rw = find_subspace_90_percent(stacked_channels_rw, 
                                                                      percent=percent, 
                                                                      k_predefined_2=k_predefined2) 
        
        p_angles, grassmann_dist = plot_subspace(dominant_subspace_rw, dominant_subspace)
        # print(f'grassmann dist (DT,RW) fine clusters: {grassmann_dist}')
        dt_subspaces.append(dominant_subspace)
        rw_subspaces.append(dominant_subspace_rw)
        dim.append(k_percent)
    
    return dt_subspaces, rw_subspaces, kmeans_centroids, kmeans_labels

#%% KMEDOID
def k_med(reduced_subspaces, 
          pos_coeff, 
          subspace_coeff, 
          kmeans_centroids, 
          n_areas, 
          kmeans_labels, 
          pos, 
          enabled_idxs, 
          bs_pos,
          seed=42): 
    
    distance_matrix = compute_distance_matrix(
        reduced_subspaces, 
        pos_coeff, 
        subspace_coeff, 
        kmeans_centroids)

    k_medoids = KMedoids(n_clusters=n_areas, random_state=seed, max_iter=300) 
    k_medoids.fit(distance_matrix)
    k_medoids_labels = k_medoids.labels_
    medoid_positions = kmeans_centroids[k_medoids.medoid_indices_]

    final_labels = np.zeros(len(pos), dtype=int)
    
    for i, label in enumerate(kmeans_labels):
        final_labels[i] = k_medoids_labels[label]

    areas = [preprocess.Area(enabled_idxs[np.where(final_labels == i)[0]], 
                      name=i, center=medoid_positions[i][:2] / pos_coeff) for i in range(n_areas)]
    
    preprocess.plot_areas(areas, pos, bs_pos)
    
    area_lens = [np.sum(final_labels == i) for i in range(n_areas)]
    min_idxs, max_idxs, mean_idxs = np.min(area_lens), np.max(area_lens), np.mean(area_lens)
    print(f'Areas have min {min_idxs} idxs, max {max_idxs} idxs, and an avg of {mean_idxs} idxs.')

    return areas, area_lens

#%% SUBSPACE-BASED CHANNEL ESTIMATION
def subspace_estimation(dataset_2paths_raw, 
                        dataset_20paths_raw, 
                        areas, 
                        area_lens, 
                        codebook,
                        n_pilots,
                        dataset_type,
                        k_predefined_2=None, 
                        snr_db=15,
                        loss_func='nmse',
                        dft_based=False,
                        seed=42):
    
    n_areas = len(areas)
    n_users = float(len(dataset_2paths_raw)) 
    nmse_ss = np.zeros((n_areas, len(n_pilots)))
    
    for selected_area_idx in range(n_areas):
        
        chs_dt = dataset_2paths_raw[areas[selected_area_idx].idxs]
        chs_rw = dataset_20paths_raw[areas[selected_area_idx].idxs]
        chs_rw_reshaped = chs_rw.view(chs_rw.shape[0], -1)  
        
        k_90 = np.max(n_pilots) # 128
        
        if dataset_type == "Random DFT-based Pilots":
            np.random.seed(seed)
            selected_columns = np.random.choice(codebook.shape[1], size=k_90, replace=False)
            U_k = codebook[:, selected_columns]
        else:
            selected_columns = majority_vote_bins_in_angle_domain(chs_dt, torch.tensor(codebook).to(torch.complex64), k=k_90)
            U_k = codebook[:, selected_columns]
        
        for n_pilot_idx, n_pilot in enumerate(n_pilots):
            
            chs_ss = project_onto_subspace(chs_rw_reshaped, U_k[:, -n_pilot:])
            estimation_subspace, noise_pow_zone = reconstruct_from_subspace(chs_ss, U_k[:, -n_pilot:], snr_db=snr_db, seed=seed)
    
            if loss_func == 'nmse':
                perf = nmse_loss(estimation_subspace, 
                                 chs_rw.view(chs_rw.shape[0], -1))
            elif loss_func == 'cosine':
                perf = torch.mean(cosine_sim(estimation_subspace, 
                                             chs_rw.view(chs_rw.shape[0], -1)))
            elif loss_func == 'throughput':
                f = estimation_subspace.conj() / torch.norm(estimation_subspace, dim=1, keepdim=True)
                snr_per_user = (torch.abs(torch.sum(chs_rw_reshaped * f.conj(), dim=1)) ** 2).unsqueeze(1) / noise_pow_zone
                perf = torch.mean(torch.log2(1 + snr_per_user))
                
            nmse_ss[selected_area_idx, n_pilot_idx] = perf.item() # n_pilot-1
            
            print(f"zone id: {selected_area_idx}, n_pilots: {n_pilot}, perf: {perf.item()}")

    nmse_ss = torch.tensor(nmse_ss, dtype=torch.float32)
    area_lens = torch.tensor(area_lens, dtype=torch.float32).view(-1, 1)  
    avg_nmse_ss = (nmse_ss.T @ area_lens) / n_users
    
    return avg_nmse_ss.numpy()

#%% DRL
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class RobustDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class SafeDRLAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # Twin Q-networks for stability
        self.online_net = RobustDQN(state_dim, action_dim).to(device)
        self.target_net = RobustDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=100000)
        
        # Adaptive exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.eps_decay = 0.9998
        
        # Performance tracking
        self.best_perf = -np.inf
        self.best_columns = None

    def act(self, state, available_actions, current_perf):
        # Epsilon-greedy with performance threshold
        if np.random.rand() <= self.epsilon or current_perf < self.best_perf:
            return np.random.choice(available_actions)
            
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.online_net(state).cpu().numpy()
            
        q_values[~np.isin(np.arange(self.action_dim), available_actions)] = -np.inf
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
        # Update best configuration
        if reward > 0:
            self.best_perf = max(self.best_perf, reward)
            self.best_columns = next_state

    def update(self, batch_size=256):
        if len(self.memory) < batch_size:
            return
            
        # Sample from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        
        # Double DQN update
        with torch.no_grad():
            target_q = rewards + 0.99 * self.target_net(next_states).max(1)[0]
            
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1))
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.eps_decay)
        
        # Soft target update
        for target_param, online_param in zip(self.target_net.parameters(), 
                                            self.online_net.parameters()):
            target_param.data.copy_(0.005 * online_param.data + 0.995 * target_param.data)

def subspace_estimation_drl(dataset_2paths_raw, dataset_20paths_raw, 
                            areas, area_lens, codebook, dataset_type, n_pilots=32, 
                            n_episodes=200, snr_db=15, loss_func='nmse', seed=42):
    
    n_users = float(len(dataset_2paths_raw)) 
    
    # Initialize agents
    agents = [SafeDRLAgent(128, 128) for _ in areas]
    results = []
    report = []
    
    # To store performance history for plotting
    perf_history_all_zones = []  # List of lists: [zone1_perf_history, zone2_perf_history, ...]
    epsilon_history_all_zones = []  # Exploration rate history
    reward_history_all_zones = []  # Reward history
    
    for zone_idx, area in enumerate(areas):
        print(f"\n--- Optimizing Zone {zone_idx+1} ---")
        
        # Load channel data
        chs_dt = dataset_2paths_raw[area.idxs]
        chs_rw = dataset_20paths_raw[area.idxs]
        chs_rw_flat = chs_rw.view(chs_rw.shape[0], -1)
        
        # Initialize selected_columns with DT-based beams
        if dataset_type == "RL-Calibrated Random Pilots":
            np.random.seed(seed)
            sorted_beams = np.random.choice(codebook.shape[1], n_pilots, False)
        elif dataset_type == "RL-Calibrated Digital Twin":
            sorted_beams = majority_vote_bins_in_angle_domain(chs_dt, codebook, k=n_pilots)
            
        current_beams = sorted_beams.copy()
        
        # Get baseline performance
        U_init = codebook[:, current_beams]
        est_init, noise = reconstruct_from_subspace(
            project_onto_subspace(chs_rw_flat, U_init), 
            U_init, snr_db, seed
        )
        base_perf = get_performance(est_init, chs_rw_flat, loss_func, noise)
        best_perf = base_perf
        print(f"Initial {loss_func}: {base_perf:.4f}")
        
        agent = agents[zone_idx]
        perf_history = [base_perf] * 10  # Initialize with baseline performance
        
        # Training loop
        zone_perf_history = []  # To store performance for this zone
        zone_epsilon_history = []  # To store exploration rate for this zone
        zone_reward_history = []  # To store rewards for this zone
        
        with tqdm(range(n_episodes)) as pbar:
            for episode in pbar:
                # Create state: beam usage + recent performance
                state = np.zeros(128)
                state[current_beams] = 1
                state[-10:] = perf_history[-10:]  # Always use the last 10 performances
                
                # Get valid actions (unused beams)
                available_actions = np.setdiff1d(np.arange(128), current_beams)
                
                # Select action with performance constraint
                action = agent.act(state, available_actions, perf_history[-1])
                
                if dataset_type == "RL-Calibrated Random Pilots":
                    replace_idx = np.random.randint(0, len(current_beams))  # Randomly select a beam to replace
                elif dataset_type == "RL-Calibrated Digital Twin":
                    # Replace the least dominant beam (first in the sorted list)
                    replace_idx = 0  # Always replace the least dominant beam
                
                new_beams = current_beams.copy()
                new_beams[replace_idx] = action
                
                # Evaluate new configuration
                U_new = codebook[:, new_beams]
                est_new, noise = reconstruct_from_subspace(
                    project_onto_subspace(chs_rw_flat, U_new), 
                    U_new, snr_db, seed
                )
                new_perf = get_performance(est_new, chs_rw_flat, loss_func, noise)
                
                # Calculate shaped reward
                perf_delta = new_perf - best_perf
                reward = np.clip(perf_delta / (abs(base_perf) + 1e-6), -1, 1)
                
                # Update agent
                next_state = np.zeros(128)
                next_state[new_beams] = 1
                next_state[-10:] = perf_history[-9:] + [new_perf]  # Update performance history
                agent.remember(state, action, reward, next_state)
                agent.update()
                
                # Update tracking
                if new_perf > best_perf:
                    best_perf = new_perf
                    current_beams = new_beams.copy()
                    
                perf_history.append(new_perf)
                zone_perf_history.append(new_perf)  # Store performance for this episode
                zone_epsilon_history.append(agent.epsilon)  # Store exploration rate
                zone_reward_history.append(reward)  # Store reward
                
                pbar.set_postfix({
                    "Best": f"{best_perf:.4f}",
                    "Current": f"{new_perf:.4f}",
                    "ε": f"{agent.epsilon:.3f}"
                })
                
        perf_history_all_zones.append(zone_perf_history)
        epsilon_history_all_zones.append(zone_epsilon_history)
        reward_history_all_zones.append(zone_reward_history)
        
        report.append({
            "initial": base_perf,
            "final": best_perf,
            "improvement": best_perf - base_perf,
            "beams": current_beams
        })
        
        results.append(best_perf)
    
    # Final Results
    results = torch.tensor(results, dtype=torch.float32)
    area_lens = torch.tensor(area_lens, dtype=torch.float32).view(-1, 1)  
    avg_nmse_ss = (results.T @ area_lens) / n_users
    
    return avg_nmse_ss

# Helper functions
def get_performance(estimation, target, metric, noise_pow=None):
    if metric == 'nmse':
        return nmse_loss(estimation, target).item()
    elif metric == 'cosine':
        return torch.mean(cosine_sim(estimation, target)).item()
    elif metric == 'throughput':
        f = estimation.conj() / torch.norm(estimation, dim=1, keepdim=True)
        snr = (torch.abs(torch.sum(target * f.conj(), 1)) ** 2) / noise_pow
        return torch.mean(torch.log2(1 + snr)).item()

#%% plot_distr
def plot_distr(dataset_rw, dataset_dt):
    
    import numpy as np
    import matplotlib.pyplot as plt

    error = dataset_rw 
    error_cov = np.zeros((error.shape[1], error.shape[1]), dtype=np.complex128)
    
    for n in range(len(error)):
        error_vector = np.expand_dims(error[n], axis=1)
        centered_error_vector = error_vector - np.mean(error_vector)
        error_cov += centered_error_vector @ centered_error_vector.conj().T
    
    error_cov = error_cov / len(error)
    eigenvalues = np.linalg.eigvals(error_cov)
    
    plt.figure(figsize=(6, 4), dpi=500)
    plt.bar(range(1, len(eigenvalues) + 1), np.sort(eigenvalues.real), color='orange', edgecolor='black', alpha=0.7)
    plt.title("Eigenvalues of the dataset", fontsize=16)
    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Eigenvalue", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

#%% GENERATE CHANNELS
def chs_gen(scenarios, n_beams, fov, n_path, codebook):
    deepmimo_data = []
    enabled_idxs = []
    dataset = []
    los_status = []
    best_beam = []
    pos = []
    bs_pos = []
    for scenario_idx, scenario in enumerate(scenarios):
        deepmimo_data.append(DeepMIMO_data_gen(scenario, n_path[scenario_idx]))
        enabled_idxs.append(np.where(deepmimo_data[-1]['user']['LoS'] != -1)[0])   
        cleaned_deepmimo_data = deepmimo_data[-1]['user']['channel']
        channels = np.squeeze(np.squeeze(np.array(cleaned_deepmimo_data), axis=1), axis=2) 
        dataset.append(torch.tensor(channels, dtype=torch.complex64))
    
    # enabled_idxs = np.unique(np.concatenate(enabled_idxs)) #############
    enabled_idxs = enabled_idxs[1]
    # print(enabled_idxs)
    for idx, scenario in enumerate(scenarios):
        los_status.append(label_gen('LoS/NLoS Classification', deepmimo_data[idx], scenario, codebook, manual_filter=enabled_idxs).reshape(-1, 1))
        best_beam.append(label_gen('Beam Prediction', deepmimo_data[idx], scenario, codebook, n_beams=n_beams, fov=fov, manual_filter=enabled_idxs).reshape(-1, 1))
        pos.append(deepmimo_data[idx]['user']['location'][enabled_idxs][:,:3])
        bs_pos.append(deepmimo_data[idx]['location'][:3])
    
    dataset_dt = dataset[0][enabled_idxs]*1e6
    dataset_rw = dataset[1][enabled_idxs]*1e6
    
    enabled_idxs = np.arange(len(dataset_rw))
    
    plot_distr(dataset_rw, dataset_dt)
    
    return dataset_dt, dataset_rw, pos[0], los_status, best_beam, enabled_idxs, bs_pos
#%% MAJORITY VOTE
from collections import Counter
def majority_vote_bins_in_angle_domain(channels, dft_codebook, k=5):
    angle_domain_channels = dft_codebook @ channels.T  
    dominant_bins = []
    for channel in angle_domain_channels.T:  
        magnitudes = np.abs(channel)
        top_bins = np.argsort(magnitudes)[-k:]  
        dominant_bins.extend(top_bins)
    bin_counts = Counter(dominant_bins)
    majority_bins = [bin for bin, _ in bin_counts.most_common(k)]
    return majority_bins

