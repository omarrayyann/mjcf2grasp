import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from assets.grippers.robotiq.robotiq_gripper import RobotiqGripper


def plot_gripper_contact_depth_2d(gripper):
    """
    Plot 2D side view of gripper (Z-X axes) showing contact depth positions from 0 to 1.
    0 = base (closer to palm), 1 = tip (furthest from palm)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get ray origins to determine contact positions
    origins, directions = gripper.get_closing_rays(np.eye(4))
    
    # Separate left and right finger rays
    left_rays = origins[::2]  # Even indices = left finger
    right_rays = origins[1::2]  # Odd indices = right finger
    
    num_rays_per_finger = len(left_rays)
    
    # Plot the finger meshes with proper triangulation
    left_finger = gripper.finger_l
    right_finger = gripper.finger_r
    
    # Get finger vertices projected onto Z-X plane
    left_verts_zx = left_finger.vertices[:, [2, 0]]  # Z, X
    right_verts_zx = right_finger.vertices[:, [2, 0]]  # Z, X
    
    # Plot finger mesh triangles
    for face in left_finger.faces:
        triangle = left_verts_zx[face]
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'b-', linewidth=0.5, alpha=0.8)
    ax.fill(left_verts_zx[:, 0], left_verts_zx[:, 1], color='steelblue', alpha=0.5, label='Left Finger')
    
    for face in right_finger.faces:
        triangle = right_verts_zx[face]
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'r-', linewidth=0.5, alpha=0.8)
    ax.fill(right_verts_zx[:, 0], right_verts_zx[:, 1], color='lightcoral', alpha=0.5, label='Right Finger')
    
    # Plot base
    base = gripper.get_base_mesh()
    base_verts_zx = base.vertices[:, [2, 0]]
    for face in base.faces:
        triangle = base_verts_zx[face]
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'k-', linewidth=0.4, alpha=0.6)
    ax.fill(base_verts_zx[:, 0], base_verts_zx[:, 1], color='gray', alpha=0.4, label='Base')
    
    # Plot contact ray positions with color gradient from 0 (base) to 1 (tip)
    cmap = plt.cm.plasma
    
    # Plot left finger contact positions
    for i, ray_origin in enumerate(left_rays):
        depth = i / (num_rays_per_finger - 1)  # Normalize to 0-1
        color = cmap(depth)
        ax.plot(ray_origin[2], ray_origin[0], 'o', color=color, markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5)
        
        # Add direction arrow
        arrow_length = 0.015
        ax.arrow(ray_origin[2], ray_origin[0], 
                arrow_length * directions[i*2][2], arrow_length * directions[i*2][0],
                head_width=0.003, head_length=0.002, fc=color, ec='black', linewidth=1)
    
    # Plot right finger contact positions
    for i, ray_origin in enumerate(right_rays):
        depth = i / (num_rays_per_finger - 1)  # Normalize to 0-1
        color = cmap(depth)
        ax.plot(ray_origin[2], ray_origin[0], 'o', color=color, markersize=10,
                markeredgecolor='black', markeredgewidth=1.5)
        
        # Add direction arrow
        arrow_length = 0.015
        ax.arrow(ray_origin[2], ray_origin[0],
                arrow_length * directions[i*2 + 1][2], arrow_length * directions[i*2 + 1][0],
                head_width=0.003, head_length=0.002, fc=color, ec='black', linewidth=1)
    
    # Add standoff range visualization
    z_min, z_max = gripper.standoff_range
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Center Line')
    ax.axvline(x=z_min, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Standoff Min ({z_min:.3f})')
    ax.axvline(x=z_max, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Standoff Max ({z_max:.3f})')
    
    # Add TCP offset marker
    if hasattr(gripper, 'tcp_offset'):
        ax.plot(gripper.tcp_offset[2], gripper.tcp_offset[0], 'b*', markersize=20, 
                markeredgecolor='black', markeredgewidth=1.5, label='TCP Offset')
    
    # Add origin marker
    ax.plot(0, 0, 'r*', markersize=20, markeredgecolor='black', 
            markeredgewidth=1.5, label='Origin (0, 0, 0)')
    
    # Add colorbar to show depth scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Contact Depth (0=Base, 1=Tip)', rotation=270, labelpad=20, fontsize=12)
    
    # Add text annotations for depth values
    # Annotate first (base) and last (tip) contact points
    ax.text(left_rays[0][2] - 0.01, left_rays[0][0], '0.0\n(Base)', 
            fontsize=10, ha='right', va='center', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(left_rays[-1][2] - 0.01, left_rays[-1][0], '1.0\n(Tip)', 
            fontsize=10, ha='right', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add label at 0.74 depth (tip center)
    tip_center_depth = 0.74
    tip_center_idx = int(tip_center_depth * (num_rays_per_finger - 1))
    if tip_center_idx < len(left_rays):
        tip_center_z = left_rays[tip_center_idx][2]
        ax.axvline(x=tip_center_z, color='purple', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(tip_center_z, ax.get_ylim()[1] * 0.85, f'{tip_center_depth:.2f}\n(Tip Center)', 
                fontsize=10, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9))
    
    # Labels and formatting
    ax.set_xlabel('Z Axis (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('X Axis (m)', fontsize=14, fontweight='bold')
    ax.set_title('Robotiq Gripper - 2D Side View (Z-X Plane)\nContact Depth Visualization', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    
    # Add information box
    info_text = (
        f'Total contact rays per finger: {num_rays_per_finger}\n'
        f'Gripper opening (q): {gripper.q:.4f} m\n'
        f'TCP Offset: [{gripper.tcp_offset[0]:.4f}, {gripper.tcp_offset[1]:.4f}, {gripper.tcp_offset[2]:.4f}] m'
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def main():
    gripper = RobotiqGripper(root_folder="")
    plot_gripper_contact_depth_2d(gripper)
    return gripper


if __name__ == "__main__":
    gripper = main()
