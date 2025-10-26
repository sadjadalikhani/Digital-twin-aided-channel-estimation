# -*- coding: utf-8 -*-
"""
Digital Twin and Real-World Channel Generation

This script generates both digital twin and real-world channel datasets using the
deepverse_comm.py framework. The datasets are formatted for PyTorch usage.

Real-World Dataset:
- Original DeepVerse dataset with 25 max paths
- Doppler effects enabled
- Represents realistic mobile communication scenarios

Digital Twin Dataset:
- Simplified dataset with 5 max paths
- Doppler effects disabled
- Represents simplified digital twin model

Output Format:
- Datasets of size (N, 128) where N is the number of channel samples
- PyTorch tensors for ML training
"""

import numpy as np
import torch
from .deepverse_comm import CommChannelGenerator
import matplotlib.pyplot as plt


def chs_gen(scenarios, n_beams, fov, n_path, codebook=None):
    """
    Generate Digital Twin and Real-World channel datasets
    
    Args:
        scenarios (list): List of scene indices to use
        n_beams (int): Number of beams (128)
        fov (float): Field of view parameter
        n_path (list): [digital_twin_paths, real_world_paths] - [5, 25]
        codebook (np.array): Optional beam steering codebook
        
    Returns:
        tuple: (dataset_dt, dataset_rw, pos, los_status, best_beam, enabled_idxs, bs_pos)
            - dataset_dt: Digital twin dataset (N, 128)
            - dataset_rw: Real-world dataset (N, 128)
            - pos: User positions
            - los_status: Line-of-sight status
            - best_beam: Best beam indices
            - enabled_idxs: Enabled user indices
            - bs_pos: Base station position
    """
    
    print("=== Digital Twin and Real-World Channel Generation ===")
    print(f"Scenarios: {scenarios}")
    print(f"Number of beams: {n_beams}")
    print(f"Paths - Digital Twin: {n_path[0]}, Real-World: {n_path[1]}")
    print()
    
    # Initialize generators for both datasets
    generator_dt = CommChannelGenerator(scenario_name="Carla-Town05")
    generator_rw = CommChannelGenerator(scenario_name="Carla-Town05")
    
    # Configure Digital Twin dataset
    print("=== Configuring Digital Twin Dataset ===")
    generator_dt.load_parameters(
        num_paths=n_path[0],  # 5 paths
        scenes=scenarios,
        enable_doppler=False  # No Doppler effects
    )
    generator_dt.generate_dataset()
    
    # Configure Real-World dataset
    print("\n=== Configuring Real-World Dataset ===")
    generator_rw.load_parameters(
        num_paths=n_path[1],  # 25 paths
        scenes=scenarios,
        enable_doppler=True   # Doppler effects enabled
    )
    generator_rw.generate_dataset()
    
    # Generate overlayed users for Real-World dataset first (as reference)
    print("\n=== Generating Overlayed Users ===")
    overlayed_data_rw = generator_rw.overlay_users_from_scenes(scenes_to_overlay=scenarios, bs_idx=0)
    
    # Extract channel data and convert to (N, 128) format
    print("\n=== Processing Channel Data ===")
    
    # Process Real-World channels first
    channels_rw = []
    valid_users = []  # Store valid user information for DT generation
    
    for i in range(overlayed_data_rw['total_users']):
        try:
            channel = generator_rw.get_overlayed_channel(i)
            
            # Debug: print actual shape
            print(f"RW user {i}: channel shape = {channel.shape}")
            
            # Ensure shape is (128,) for (N, 128) dataset
            if channel.shape == (128, 1):
                channel = channel.flatten()
            elif channel.shape == (1, 128):
                channel = channel.flatten()
            elif len(channel.shape) == 2:
                # Handle any 2D shape by flattening
                channel = channel.flatten()
            channels_rw.append(channel)
            
            # Store valid user info for DT generation
            valid_users.append({
                'scene_idx': overlayed_data_rw['scene_ids'][i],
                'ue_idx': overlayed_data_rw['user_ids'][i],
                'location': overlayed_data_rw['locations'][i]
            })
        except Exception as e:
            print(f"Warning: Could not process RW user {i}: {e}")
            continue
    
    # Check if we have any valid users
    if len(valid_users) == 0:
        raise ValueError("No valid users were processed from Real-World dataset. Check channel shapes.")
    
    print(f"Generating {len(valid_users)} corresponding Digital Twin channels...")
    channels_dt = []
    
    for i, user_info in enumerate(valid_users):
        try:
            # Get DT channel for the same user (scene_idx, ue_idx)
            channel_sample = generator_dt.get_bs_ue_channel(
                scene_idx=user_info['scene_idx'], 
                bs_idx=0, 
                ue_idx=user_info['ue_idx']
            )
            channel = channel_sample.coeffs
            
            # Handle None or empty channels (no paths to BS)
            if channel is None or channel.size == 0:
                print(f"Warning: DT user {i} (Scene {user_info['scene_idx']}, UE {user_info['ue_idx']}) has no paths - using zero channel")
                # Create zero channel with correct shape (128,)
                channel = np.zeros(128, dtype=complex)
            else:
                # Ensure shape is (128,) for (N, 128) dataset
                if len(channel.shape) == 3:
                    channel = channel.squeeze(-1)  # Remove subcarrier dimension
                if channel.shape == (128, 1):
                    channel = channel.flatten()
                elif channel.shape == (1, 128):
                    channel = channel.flatten()
            
            channels_dt.append(channel)
            
        except Exception as e:
            print(f"Warning: Could not process DT user {i} (Scene {user_info['scene_idx']}, UE {user_info['ue_idx']}): {e}")
            print(f"Using zero channel to maintain pairing")
            # Create zero channel to maintain pairing
            channel = np.zeros(128, dtype=complex)
            channels_dt.append(channel)
    
    # Convert to numpy arrays and then to PyTorch tensors
    dataset_dt = torch.tensor(np.array(channels_dt), dtype=torch.complex64)
    dataset_rw = torch.tensor(np.array(channels_rw), dtype=torch.complex64)
    
    # Extract additional information using valid_users
    pos = np.array([user['location'] for user in valid_users])  # User positions
    bs_pos = generator_rw.get_locations(scene_idx=0, bs_idx=0, ue_idx=0)[0]  # BS position
    
    # Get LoS status for each valid user
    los_status = []
    enabled_idxs = []
    for i, user_info in enumerate(valid_users):
        try:
            # Get channel sample to extract LoS status
            channel_sample = generator_rw.get_bs_ue_channel(
                scene_idx=user_info['scene_idx'], 
                bs_idx=0, 
                ue_idx=user_info['ue_idx']
            )
            los_status.append(channel_sample.LoS_status)
            enabled_idxs.append(i)
        except Exception as e:
            print(f"Warning: Could not get LoS status for user {i}: {e}")
            continue
    
    los_status = np.array(los_status)
    enabled_idxs = np.array(enabled_idxs)
    
    # Calculate best beam indices using beam steering
    if codebook is not None:
        best_beam = calculate_best_beams(dataset_rw, codebook)
    else:
        # Generate default codebook if none provided
        codebook = generate_default_codebook(n_beams)
        best_beam = calculate_best_beams(dataset_rw, codebook)
    
    # Verify that both datasets have the same number of samples
    if dataset_dt.shape[0] != dataset_rw.shape[0]:
        raise ValueError(f"Dataset size mismatch: DT has {dataset_dt.shape[0]} samples, RW has {dataset_rw.shape[0]} samples")
    
    print(f"\n=== Dataset Summary ===")
    print(f"Digital Twin dataset shape: {dataset_dt.shape}")
    print(f"Real-World dataset shape: {dataset_rw.shape}")
    print(f"✓ Both datasets have {dataset_dt.shape[0]} samples (perfectly paired)")
    print(f"User positions shape: {pos.shape}")
    print(f"LoS status shape: {los_status.shape}")
    print(f"Best beam indices shape: {best_beam.shape}")
    print(f"Enabled user indices: {len(enabled_idxs)}")
    print(f"Base station position: {bs_pos}")
    
    return dataset_dt, dataset_rw, pos, los_status, best_beam, enabled_idxs, bs_pos


def generate_default_codebook(n_beams=128):
    """
    Generate default beam steering codebook
    
    Args:
        n_beams (int): Number of beams
        
    Returns:
        np.array: Beam steering codebook of shape (n_beams, 128)
    """
    from .deepverse_comm import beam_steering_codebook
    
    # Generate beam angles
    x_angles = np.linspace(0, 180, n_beams + 1)[1:]
    x_angles = np.flip(x_angles)
    z_angles = np.full(n_beams, 90)
    beam_angles = np.column_stack((z_angles, x_angles))
    
    # Generate codebook
    codebook = beam_steering_codebook(beam_angles, 128, 1)
    
    return codebook


def calculate_best_beams(channels, codebook):
    """
    Calculate best beam indices for each channel
    
    Args:
        channels (torch.Tensor): Channel dataset of shape (N, 128)
        codebook (np.array): Beam steering codebook of shape (n_beams, 128)
        
    Returns:
        np.array: Best beam indices for each channel
    """
    # Convert to numpy for computation
    channels_np = channels.numpy()
    
    # Calculate beam power for each channel
    beam_power = np.abs(codebook @ channels_np.T)**2  # (n_beams, N)
    
    # Find best beam for each channel
    best_beam = np.argmax(beam_power, axis=0)
    
    return best_beam


def visualize_datasets(dataset_dt, dataset_rw, pos, bs_pos, save_plots=True):
    """
    Visualize the generated datasets
    
    Args:
        dataset_dt (torch.Tensor): Digital twin dataset
        dataset_rw (torch.Tensor): Real-world dataset
        pos (np.array): User positions
        bs_pos (np.array): Base station position
        save_plots (bool): Whether to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: User positions
    ax1 = axes[0, 0]
    ax1.scatter(bs_pos[0], bs_pos[1], color='red', s=200, marker='^', 
               edgecolors='black', linewidth=2)
    ax1.scatter(pos[:, 0], pos[:, 1], color='blue', s=30, alpha=0.6)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('User Positions')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Channel magnitude comparison
    ax2 = axes[0, 1]
    dt_magnitude = torch.abs(dataset_dt).mean(dim=0).numpy()
    rw_magnitude = torch.abs(dataset_rw).mean(dim=0).numpy()
    ax2.plot(dt_magnitude, label='Digital Twin', alpha=0.8)
    ax2.plot(rw_magnitude, label='Real-World', alpha=0.8)
    ax2.set_xlabel('Antenna Index')
    ax2.set_ylabel('Average Channel Magnitude')
    ax2.set_title('Channel Magnitude Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Channel power distribution
    ax3 = axes[1, 0]
    dt_power = torch.abs(dataset_dt)**2
    rw_power = torch.abs(dataset_rw)**2
    ax3.hist(dt_power.flatten().numpy(), bins=50, alpha=0.6, label='Digital Twin', density=True)
    ax3.hist(rw_power.flatten().numpy(), bins=50, alpha=0.6, label='Real-World', density=True)
    ax3.set_xlabel('Channel Power')
    ax3.set_ylabel('Density')
    ax3.set_title('Channel Power Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Dataset statistics
    ax4 = axes[1, 1]
    stats_data = [
        ['Digital Twin', dataset_dt.shape[0], dataset_dt.shape[1], 5, False],
        ['Real-World', dataset_rw.shape[0], dataset_rw.shape[1], 25, True]
    ]
    
    ax4.axis('tight')
    ax4.axis('off')
    table = ax4.table(cellText=stats_data,
                     colLabels=['Dataset', 'Samples', 'Antennas', 'Paths', 'Doppler'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Dataset Statistics')
    
    plt.tight_layout()
    
    if save_plots:
        filename = "figs/dt_rw_dataset_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as: {filename}")
        plt.close()
    else:
        plt.savefig("figs/dt_rw_dataset_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function demonstrating DT and RW channel generation"""
    
    # Configuration parameters
    scenarios = np.arange(10)  # Use first 100 scenes
    n_beams = 128
    fov = 180  # Field of view
    n_path = [5, 25]  # [Digital Twin paths, Real-World paths]
    
    print("=== Digital Twin and Real-World Channel Generation ===")
    print(f"Scenarios: {len(scenarios)} scenes")
    print(f"Digital Twin: {n_path[0]} paths, no Doppler")
    print(f"Real-World: {n_path[1]} paths, with Doppler")
    print()
    
    # Generate datasets
    dataset_dt, dataset_rw, pos, los_status, best_beam, enabled_idxs, bs_pos = chs_gen(
        scenarios=scenarios,
        n_beams=n_beams,
        fov=fov,
        n_path=n_path,
        codebook=None  # Will generate default codebook
    )
    
    # Visualize datasets
    print("\n=== Generating Visualizations ===")
    visualize_datasets(dataset_dt, dataset_rw, pos, bs_pos, save_plots=True)
    
    # # Save datasets
    # print("\n=== Saving Datasets ===")
    # torch.save(dataset_dt, 'figs/dataset_digital_twin.pt')
    # torch.save(dataset_rw, 'figs/dataset_real_world.pt')
    # np.save('figs/user_positions.npy', pos)
    # np.save('figs/los_status.npy', los_status)
    # np.save('figs/best_beam_indices.npy', best_beam)
    # np.save('figs/enabled_indices.npy', enabled_idxs)
    # np.save('figs/bs_position.npy', bs_pos)
    
    # print("Datasets saved:")
    # print("- figs/dataset_digital_twin.pt")
    # print("- figs/dataset_real_world.pt")
    # print("- figs/user_positions.npy")
    # print("- figs/los_status.npy")
    # print("- figs/best_beam_indices.npy")
    # print("- figs/enabled_indices.npy")
    # print("- figs/bs_position.npy")
    
    print("\n=== Generation Complete ===")
    print("Both Digital Twin and Real-World datasets are ready for ML training!")
    print(f"Dataset shapes: DT {dataset_dt.shape}, RW {dataset_rw.shape}")


if __name__ == "__main__":
    main()