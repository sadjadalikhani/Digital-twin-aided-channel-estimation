# -*- coding: utf-8 -*-
"""
Simplified DeepVerse Communication Channel Tutorial

This script focuses only on communication channel generation with tunable parameters.
Camera, LiDAR, and radar functionalities are disabled for simplicity.

Features:
- Tunable number of communication paths
- Tunable scene selection
- BS-UE and BS-BS channel generation
- Beam steering visualization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from deepverse import ParameterManager
from deepverse.scenario import ScenarioManager
from deepverse import Dataset


class CommChannelGenerator:
    """Simplified DeepVerse communication channel generator"""
    
    def __init__(self, scenario_name="Carla-Town05", config_path=None):
        """
        Initialize the communication channel generator
        
        Args:
            scenario_name (str): Name of the scenario to use
            config_path (str): Path to configuration file (optional)
        """
        self.scenario_name = scenario_name
        if config_path is None:
            self.config_path = f"scenarios/{scenario_name}/param/config.m"
        else:
            self.config_path = config_path
            
        self.param_manager = None
        self.dataset = None
        self.overlayed_users = None  # Store overlayed user data
        
    def load_parameters(self, num_paths=25, scenes=None, enable_doppler=False):
        """
        Load and configure parameters for communication channel generation
        
        Args:
            num_paths (int): Number of communication paths to generate
            scenes (list): List of scene indices to process (default: [0, 1])
            enable_doppler (bool): Whether to enable Doppler effects (default: False)
        """
        # Initialize ParameterManager and load parameters
        self.param_manager = ParameterManager(self.config_path)
        params = self.param_manager.get_params()
        
        # Set default scenes if not provided
        if scenes is None:
            scenes = [0, 1]
            
        # Configure parameters for communication channels only
        self.param_manager.params["scenes"] = scenes
        self.param_manager.params["comm"]["num_paths"] = num_paths
        self.param_manager.params["comm"]["enable"] = True
        self.param_manager.params["comm"]["enable_Doppler"] = 1 if enable_doppler else 0
        
        # Disable other modalities
        self.param_manager.params["camera"] = False
        self.param_manager.params["lidar"] = False
        self.param_manager.params["radar"]["enable"] = False
        self.param_manager.params["position"] = False
        
        print(f"Loaded parameters for {len(scenes)} scenes with {num_paths} paths each")
        print(f"Scenes: {scenes}")
        print(f"Communication enabled: {self.param_manager.params['comm']['enable']}")
        print(f"Doppler effects enabled: {enable_doppler}")
        
    def generate_dataset(self):
        """Generate the DeepVerse dataset based on configured parameters"""
        if self.param_manager is None:
            raise ValueError("Parameters not loaded. Call load_parameters() first.")
            
        print("Generating dataset...")
        self.dataset = Dataset(self.param_manager)
        print("Dataset generation completed!")
        
    def get_bs_ue_channel(self, scene_idx=0, bs_idx=0, ue_idx=0):
        """
        Get BS-UE communication channel
        
        Args:
            scene_idx (int): Scene index
            bs_idx (int): Base station index
            ue_idx (int): User equipment index
            
        Returns:
            Channel sample with coefficients and metadata
        """
        if self.dataset is None:
            raise ValueError("Dataset not generated. Call generate_dataset() first.")
            
        return self.dataset.get_sample('comm-ue', index=scene_idx, bs_idx=bs_idx, ue_idx=ue_idx)
    
    def get_bs_bs_channel(self, scene_idx=0, bs_idx=0, ue_idx=0):
        """
        Get BS-BS communication channel
        
        Args:
            scene_idx (int): Scene index
            bs_idx (int): Base station index
            ue_idx (int): User equipment index (used as second BS index)
            
        Returns:
            Channel sample with coefficients and metadata
        """
        if self.dataset is None:
            raise ValueError("Dataset not generated. Call generate_dataset() first.")
            
        return self.dataset.get_sample('comm-bs', index=scene_idx, bs_idx=bs_idx, ue_idx=ue_idx)
    
    def get_locations(self, scene_idx=0, bs_idx=0, ue_idx=0):
        """
        Get BS and UE locations
        
        Args:
            scene_idx (int): Scene index
            bs_idx (int): Base station index
            ue_idx (int): User equipment index
            
        Returns:
            tuple: (bs_location, ue_location)
        """
        if self.dataset is None:
            raise ValueError("Dataset not generated. Call generate_dataset() first.")
            
        bs_location = self.dataset.get_sample('loc-bs', index=scene_idx, bs_idx=bs_idx)
        ue_location = self.dataset.get_sample('loc-ue', index=scene_idx, ue_idx=ue_idx)
        
        return bs_location, ue_location
    
    def overlay_users_from_scenes(self, scenes_to_overlay=None, bs_idx=0):
        """
        Overlay users from multiple scenes to create a larger candidate user set
        
        Args:
            scenes_to_overlay (list): List of scene indices to overlay (default: all loaded scenes)
            bs_idx (int): Base station index
            
        Returns:
            dict: Dictionary containing overlayed user data
        """
        if self.dataset is None:
            raise ValueError("Dataset not generated. Call generate_dataset() first.")
            
        if scenes_to_overlay is None:
            scenes_to_overlay = self.param_manager.params['scenes']
            
        overlayed_data = {
            'channels': [],
            'locations': [],
            'scene_ids': [],
            'user_ids': [],
            'total_users': 0
        }
        
        print(f"Overlaying users from {len(scenes_to_overlay)} scenes...")
        
        for scene_idx in scenes_to_overlay:
            # Get all users in this scene (assuming 8 users per scene based on file structure)
            for ue_idx in range(8):  # Users 0-7
                try:
                    # Get channel and location for this user
                    channel = self.get_bs_ue_channel(scene_idx=scene_idx, bs_idx=bs_idx, ue_idx=ue_idx)
                    _, ue_location = self.get_locations(scene_idx=scene_idx, bs_idx=bs_idx, ue_idx=ue_idx)
                    
                    # Check if channel data is valid
                    if channel.coeffs is not None and channel.coeffs.size > 0:
                        # Store the data
                        overlayed_data['channels'].append(channel.coeffs)
                        overlayed_data['locations'].append(ue_location)
                        overlayed_data['scene_ids'].append(scene_idx)
                        overlayed_data['user_ids'].append(ue_idx)
                        overlayed_data['total_users'] += 1
                    else:
                        print(f"Warning: Empty channel data for user {ue_idx} from scene {scene_idx}")
                        continue
                    
                except Exception as e:
                    print(f"Warning: Could not load user {ue_idx} from scene {scene_idx}: {e}")
                    continue
        
        self.overlayed_users = overlayed_data
        print(f"Successfully overlayed {overlayed_data['total_users']} users from {len(scenes_to_overlay)} scenes")
        
        return overlayed_data
    
    def get_overlayed_channel(self, user_index):
        """
        Get channel for a specific overlayed user
        
        Args:
            user_index (int): Index in the overlayed user list
            
        Returns:
            np.array: Channel coefficients of shape (128, 1)
        """
        if self.overlayed_users is None:
            raise ValueError("No overlayed users available. Call overlay_users_from_scenes() first.")
            
        if user_index >= self.overlayed_users['total_users']:
            raise ValueError(f"User index {user_index} out of range. Total users: {self.overlayed_users['total_users']}")
            
        # Return channel in (128, 1) format
        channel = self.overlayed_users['channels'][user_index]
        
        # Handle None channels
        if channel is None:
            raise ValueError(f"Channel data is None for user {user_index}")
        
        # Debug: print raw channel shape
        print(f"DEBUG: Raw channel shape for user {user_index}: {channel.shape}")
        
        # Ensure channel has the expected shape (128, 1)
        if len(channel.shape) == 3:
            # Remove the subcarrier dimension if present: (1, 128, 1) -> (1, 128)
            # Only squeeze if the last dimension is 1
            if channel.shape[-1] == 1:
                channel = channel.squeeze(-1)
            elif channel.shape[0] == 1:
                channel = channel.squeeze(0)
        
        if len(channel.shape) == 2 and channel.shape[0] == 1:
            # Transpose to get (128, 1) from (1, 128)
            channel = channel.T
        elif len(channel.shape) == 2 and channel.shape[1] == 1:
            # Already in (128, 1) format
            pass
        elif len(channel.shape) == 1:
            # Already 1D, reshape to (128, 1)
            channel = channel.reshape(-1, 1)
        else:
            # Try to reshape to (128, 1)
            try:
                channel = channel.reshape(-1, 1)
            except:
                print(f"ERROR: Cannot reshape channel of shape {channel.shape} to (128, 1)")
                raise
            
        return channel
    
    def get_overlayed_location(self, user_index):
        """
        Get location for a specific overlayed user
        
        Args:
            user_index (int): Index in the overlayed user list
            
        Returns:
            np.array: User location
        """
        if self.overlayed_users is None:
            raise ValueError("No overlayed users available. Call overlay_users_from_scenes() first.")
            
        if user_index >= self.overlayed_users['total_users']:
            raise ValueError(f"User index {user_index} out of range. Total users: {self.overlayed_users['total_users']}")
            
        return self.overlayed_users['locations'][user_index]


def beam_steering_codebook(angles, num_z, num_x):
    """
    Generate beam steering codebook
    
    Args:
        angles (np.array): Array of beam angles [z_angle, x_angle]
        num_z (int): Number of antennas in z direction
        num_x (int): Number of antennas in x direction
        
    Returns:
        np.array: Beamforming codebook
    """
    d = 0.5  # Antenna spacing
    k_z = np.arange(num_z)
    k_x = np.arange(num_x)

    codebook = []

    for beam_idx in range(angles.shape[0]):
        z_angle = angles[beam_idx, 0]
        x_angle = angles[beam_idx, 1]
        bf_vector_z = np.exp(1j * 2 * np.pi * k_z * d * np.cos(np.radians(z_angle)))
        bf_vector_x = np.exp(1j * 2 * np.pi * k_x * d * np.cos(np.radians(x_angle)))
        
        # Handle different antenna configurations
        if num_z == 1 and num_x > 1:
            # Linear array in x direction
            bf_vector = bf_vector_x
        elif num_z > 1 and num_x == 1:
            # Linear array in z direction
            bf_vector = bf_vector_z
        else:
            # 2D array
            bf_vector = np.outer(bf_vector_z, bf_vector_x).flatten()
            
        codebook.append(bf_vector)

    return np.stack(codebook, axis=0)


def analyze_beam_power(generator, num_angles=64, bs_idx=0, ue_idx=0):
    """
    Analyze beam power across scenes
    
    Args:
        generator (CommChannelGenerator): Channel generator instance
        num_angles (int): Number of beam angles to test
        bs_idx (int): Base station index
        ue_idx (int): User equipment index
        
    Returns:
        tuple: (beam_power, ue_locations, bs_location)
    """
    # Get channel shape to determine antenna configuration
    sample_channel = generator.get_bs_ue_channel(scene_idx=0, bs_idx=bs_idx, ue_idx=ue_idx)
    channel_shape = sample_channel.coeffs.shape
    num_bs_antennas = channel_shape[1]  # BS antennas are in dimension 1
    
    # Construct beam steering codebook matching the actual antenna configuration
    x_angles = np.linspace(0, 180, num_angles + 1)[1:]
    x_angles = np.flip(x_angles)
    z_angles = np.full(num_angles, 90)
    beam_angles = np.column_stack((z_angles, x_angles))
    
    # Use the actual antenna configuration: [128, 1] from config.m
    # num_bs_antennas = 128, so we have 128 antennas in z direction, 1 in x direction
    codebook = beam_steering_codebook(beam_angles, num_bs_antennas, 1)
    
    # Get number of scenes
    num_scenes = len(generator.param_manager.params['scenes'])
    
    beam_power = []
    ue_locations = []
    
    print(f"Analyzing beam power for {num_scenes} scenes...")
    
    for i in range(num_scenes):
        # Get channel and UE location
        channel = generator.get_bs_ue_channel(scene_idx=i, bs_idx=bs_idx, ue_idx=ue_idx)
        ue_location = generator.get_locations(scene_idx=i, bs_idx=bs_idx, ue_idx=ue_idx)[1]
        
        # Calculate beam power
        beam_power_ = (np.abs(codebook @ np.squeeze(channel.coeffs, 0))**2).sum(-1)
        beam_power.append(beam_power_)
        ue_locations.append(ue_location)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_scenes} scenes")
    
    # Get BS location (same for all scenes)
    bs_location = generator.get_locations(scene_idx=0, bs_idx=bs_idx, ue_idx=ue_idx)[0]
    
    return beam_power, ue_locations, bs_location


def plot_beam_power_animation(beam_power, ue_locations, bs_location, num_angles=64):
    """
    Create animated plot of beam power and UE position
    
    Args:
        beam_power (list): List of beam power arrays for each scene
        ue_locations (list): List of UE locations for each scene
        bs_location (np.array): BS location
        num_angles (int): Number of beam angles
    """
    num_scenes = len(beam_power)
    
    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Initialize plots
    axes[0].scatter(bs_location[0], bs_location[1], color='b', s=100, label='BS', marker='^')
    ue_scatter = axes[0].scatter([0], [0], color='g', s=50, label='UE', marker='o')
    axes[0].set_xlim([-100, 100])
    axes[0].set_ylim([0, 120])
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_title('BS and UE Positions')
    axes[0].legend()
    axes[0].grid(True)
    
    line, = axes[1].plot(range(1, num_angles + 1), 10 * np.log10(beam_power[0]))
    axes[1].set_xlim([1, num_angles])
    axes[1].set_ylim([-120, -80])
    axes[1].set_xlabel('Beam index')
    axes[1].set_ylabel('Beam power (dB)')
    axes[1].set_title('Beam Power vs Beam Index')
    axes[1].grid(True)
    
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.15)
    
    # Update function for animation
    def update(i):
        ue_scatter.set_offsets([ue_locations[i][0], ue_locations[i][1]])
        line.set_data(range(1, num_angles + 1), 10 * np.log10(beam_power[i]))
        return ue_scatter, line
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=num_scenes, interval=200, blit=True)
    
    return ani


def plot_overlayed_users(generator, bs_idx=0):
    """
    Plot the positions of all overlayed users from multiple scenes
    
    Args:
        generator (CommChannelGenerator): Channel generator instance
        bs_idx (int): Base station index
    """
    if generator.overlayed_users is None:
        raise ValueError("No overlayed users available. Call overlay_users_from_scenes() first.")
    
    # Get BS location
    bs_location = generator.get_locations(scene_idx=0, bs_idx=bs_idx, ue_idx=0)[0]
    
    # Extract user locations and scene information
    user_locations = generator.overlayed_users['locations']
    scene_ids = generator.overlayed_users['scene_ids']
    user_ids = generator.overlayed_users['user_ids']
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot BS location
    ax.scatter(bs_location[0], bs_location[1], color='red', s=200, marker='^', 
               edgecolors='black', linewidth=2)
    
    # Get unique scene IDs for color mapping
    unique_scenes = list(set(scene_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_scenes)))
    
    # Plot users from each scene with different colors
    for i, scene_id in enumerate(unique_scenes):
        scene_mask = [s == scene_id for s in scene_ids]
        scene_locations = [loc for j, loc in enumerate(user_locations) if scene_mask[j]]
        
        x_coords = [loc[0] for loc in scene_locations]
        y_coords = [loc[1] for loc in scene_locations]
        
        ax.scatter(x_coords, y_coords, color=colors[i], s=50, alpha=0.7, marker='o')
    
    # Set plot properties
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Overlayed Users from Multiple Scenes\nTotal Users: {generator.overlayed_users["total_users"]}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save the figure instead of showing it
    filename = f"figs/overlayed_users_positions.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {filename}")
    plt.close()  # Close the figure to free memory
    
    # Print summary statistics
    print(f"\n=== Overlayed Users Summary ===")
    print(f"Total users: {generator.overlayed_users['total_users']}")
    print(f"Number of scenes: {len(unique_scenes)}")
    print(f"Average users per scene: {generator.overlayed_users['total_users'] / len(unique_scenes):.1f}")
    print(f"Scene distribution: {dict(zip(unique_scenes, [scene_ids.count(s) for s in unique_scenes]))}")


def analyze_overlayed_beam_power(generator, num_angles=64, bs_idx=0, sample_users=None):
    """
    Analyze beam power for overlayed users
    
    Args:
        generator (CommChannelGenerator): Channel generator instance
        num_angles (int): Number of beam angles to test
        bs_idx (int): Base station index
        sample_users (list): List of user indices to analyze (default: first 10 users)
        
    Returns:
        tuple: (beam_power_matrix, user_locations, bs_location)
    """
    if generator.overlayed_users is None:
        raise ValueError("No overlayed users available. Call overlay_users_from_scenes() first.")
    
    if sample_users is None:
        # Analyze first 10 users by default
        sample_users = list(range(min(10, generator.overlayed_users['total_users'])))
    
    # Get BS location
    bs_location = generator.get_locations(scene_idx=0, bs_idx=bs_idx, ue_idx=0)[0]
    
    # Construct beam steering codebook
    x_angles = np.linspace(0, 180, num_angles + 1)[1:]
    x_angles = np.flip(x_angles)
    z_angles = np.full(num_angles, 90)
    beam_angles = np.column_stack((z_angles, x_angles))
    codebook = beam_steering_codebook(beam_angles, 128, 1)  # 128 antennas in z direction
    
    # Ensure codebook has the right shape for matrix multiplication with (128, 1) channels
    # codebook should be (num_angles, 128) to multiply with (128, 1) channel
    if codebook.shape[1] != 128:
        print(f"Warning: Codebook shape {codebook.shape} doesn't match expected (num_angles, 128)")
        # Reshape if needed
        codebook = codebook.reshape(num_angles, -1)
    
    beam_power_matrix = []
    user_locations = []
    
    print(f"Analyzing beam power for {len(sample_users)} overlayed users...")
    
    for i, user_idx in enumerate(sample_users):
        try:
            # Get channel and location for this user
            channel = generator.get_overlayed_channel(user_idx)
            location = generator.get_overlayed_location(user_idx)
            
            # Debug: Print shapes for first user
            if i == 0:
                print(f"Debug - Codebook shape: {codebook.shape}, Channel shape: {channel.shape}")
            
            # Calculate beam power: codebook (num_angles, 128) @ channel (128, 1) = (num_angles, 1)
            beam_power_ = (np.abs(codebook @ channel)**2).sum(-1)
            beam_power_matrix.append(beam_power_)
            user_locations.append(location)
            
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(sample_users)} users")
                
        except Exception as e:
            print(f"Warning: Could not process user {user_idx}: {e}")
            # Debug: Print shapes when error occurs
            try:
                channel = generator.get_overlayed_channel(user_idx)
                print(f"Debug - Error user {user_idx}: Codebook shape: {codebook.shape}, Channel shape: {channel.shape}")
            except:
                pass
            continue
    
    return np.array(beam_power_matrix), user_locations, bs_location


def main():
    """Main function demonstrating communication channel generation with user overlay"""
    
    # Configuration parameters
    SCENARIO_NAME = "Carla-Town05"
    NUM_PATHS = 25  # Tunable: number of communication paths
    SCENES = np.arange(4000)  # Tunable: scene selection (changed from [100] for overlay demo)
    ENABLE_DOPPLER = False  # Tunable: enable Doppler effects for mobile scenarios
    
    print("=== DeepVerse Communication Channel Tutorial with User Overlay ===")
    print(f"Scenario: {SCENARIO_NAME}")
    print(f"Number of paths: {NUM_PATHS}")
    print(f"Scenes: {SCENES}")
    print(f"Doppler effects: {ENABLE_DOPPLER}")
    print()
    
    # Initialize generator
    generator = CommChannelGenerator(scenario_name=SCENARIO_NAME)
    
    # Load parameters with tunable settings
    generator.load_parameters(num_paths=NUM_PATHS, scenes=SCENES, enable_doppler=ENABLE_DOPPLER)
    
    # Generate dataset
    generator.generate_dataset()
    
    # Example: Get a single channel sample
    print("\n=== Channel Sample Analysis ===")
    channel_sample = generator.get_bs_ue_channel(scene_idx=0, bs_idx=0, ue_idx=0)
    print(f"Channel shape: {channel_sample.coeffs.shape}")
    print(f"Number of paths: {channel_sample.paths.num_paths() if hasattr(channel_sample, 'paths') else 'N/A'}")
    print(f"LoS status: {channel_sample.LoS_status if hasattr(channel_sample, 'LoS_status') else 'N/A'}")
    
    # Example: Get locations
    bs_loc, ue_loc = generator.get_locations(scene_idx=0, bs_idx=0, ue_idx=0)
    print(f"BS location: {bs_loc}")
    print(f"UE location: {ue_loc}")
    
    # Overlay users from multiple scenes
    print("\n=== User Overlay Analysis ===")
    overlayed_data = generator.overlay_users_from_scenes(scenes_to_overlay=SCENES, bs_idx=0)
    
    # Plot overlayed user positions
    print("\n=== Plotting Overlayed User Positions ===")
    plot_overlayed_users(generator, bs_idx=0)
    
    # Example: Access overlayed user data
    print("\n=== Overlayed User Channel Examples ===")
    valid_users = 0
    for i in range(min(3, overlayed_data['total_users'])):
        try:
            channel = generator.get_overlayed_channel(i)
            location = generator.get_overlayed_location(i)
            scene_id = overlayed_data['scene_ids'][i]
            user_id = overlayed_data['user_ids'][i]
            print(f"User {i}: Scene {scene_id}, UE {user_id}, Channel shape: {channel.shape}, Location: {location}")
            valid_users += 1
        except Exception as e:
            print(f"User {i}: Error accessing data - {e}")
            continue
    
    # Debug: Test beam steering codebook generation
    print("\n=== Debug: Beam Steering Codebook ===")
    x_angles = np.linspace(0, 180, 65)[1:]  # 64 angles
    x_angles = np.flip(x_angles)
    z_angles = np.full(64, 90)
    beam_angles = np.column_stack((z_angles, x_angles))
    test_codebook = beam_steering_codebook(beam_angles, 128, 1)
    print(f"Test codebook shape: {test_codebook.shape}")
    print(f"Expected: (64, 128), Got: {test_codebook.shape}")
    
    # Analyze beam power for overlayed users
    print("\n=== Overlayed Beam Power Analysis ===")
    beam_power_matrix, user_locations, bs_location = analyze_overlayed_beam_power(
        generator, sample_users=list(range(min(5, overlayed_data['total_users'])))
    )
    
    # Plot beam power for sample users
    print("\n=== Plotting Beam Power for Sample Users ===")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(5, len(beam_power_matrix))):
        axes[i].plot(range(1, len(beam_power_matrix[i]) + 1), 10 * np.log10(beam_power_matrix[i]))
        axes[i].set_title(f'User {i}')
        axes[i].set_xlabel('Beam Index')
        axes[i].set_ylabel('Beam Power (dB)')
        axes[i].grid(True)
    
    # Hide unused subplots
    for i in range(5, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure instead of showing it
    filename = f"figs/beam_power_sample_users.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {filename}")
    plt.close()  # Close the figure to free memory
    
    print("\nTutorial completed successfully!")
    print("Key features demonstrated:")
    print("- Single subcarrier configuration (128,1) channel shape")
    print("- User overlay from multiple scenes for ML training")
    print("- Visualization of overlayed user positions")
    print("- Beam power analysis for multiple users")
    print("- Configurable Doppler effects for mobile scenarios")
    print("\nYou can modify NUM_PATHS, SCENES, and ENABLE_DOPPLER variables to experiment with different configurations.")


if __name__ == "__main__":
    main()