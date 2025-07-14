import matplotlib
matplotlib.use('TkAgg') 
import numpy as np
import trimesh
import trimesh.transformations as tra
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PandaGripper:
    """An object representing a Franka Panda gripper."""

    def __init__(self, q=0.04, num_contact_points_per_finger=10, root_folder=''):
        self.q = q
        fn_base = root_folder + 'assets/gripper_models/rum_gripper/meshes/simple_body.stl'
        fn_finger_l = root_folder + 'assets/gripper_models/rum_gripper/meshes/simple_left.stl'
        fn_finger_r = root_folder + 'assets/gripper_models/rum_gripper/meshes/simple_right.stl'
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger_l)
        self.finger_r = trimesh.load(fn_finger_r)

        self.standoff_range = np.array([
            max(self.finger_l.bounding_box.bounds[0, 2],
                self.base.bounding_box.bounds[1, 2]),
            self.finger_l.bounding_box.bounds[1, 2]
        ])
        self.standoff_range[0] += 0.001

        self.ray_origins = []
        self.ray_directions = []

        for i in np.linspace(-0.01, 0.045, num_contact_points_per_finger):
            self.ray_origins.append(np.r_[self.finger_l.bounding_box.centroid + [0, 0, i], 1] + [i/np.sqrt(2) + 0.01, 0, 0, 1])
            self.ray_origins.append(np.r_[self.finger_r.bounding_box.centroid + [0, 0, i], 1] - [i/np.sqrt(2) + 0.01, 0, 0, 1])
            self.ray_directions.append(np.r_[-self.finger_l.bounding_box.primitive.transform[:3, 0]])
            self.ray_directions.append(np.r_[+self.finger_r.bounding_box.primitive.transform[:3, 0]])

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

    def get_finger_meshes(self):
        return [self.finger_l, self.finger_r]

    def get_base_mesh(self):
        return self.base

    def get_base_obb(self):
        return self.base.bounding_box
    
    def get_obbs(self):
        return [self.finger_l.bounding_box, self.finger_r.bounding_box]

    def get_closing_rays(self):
        return self.ray_origins[:, :3], self.ray_directions

def plot_gripper(gripper):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot finger meshes
    for mesh in gripper.get_finger_meshes():
        ax.plot_trisurf(
            mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
            triangles=mesh.faces,
            color='steelblue', alpha=0.5, edgecolor='k', linewidth=0.1
        )

    # Plot rays as arrows
    origins, directions = gripper.get_closing_rays()
    arrow_length = 0.02
    ax.quiver(
        origins[:, 0], origins[:, 1], origins[:, 2],         # ray start (x, y, z)
        directions[:, 0], directions[:, 1], directions[:, 2],# ray direction (dx, dy, dz)
        length=arrow_length, normalize=True, color='red', linewidth=1
    )

    ax.set_title("Panda Gripper Closing Rays and Finger OBBs")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz([-0.1, 0.1], [-0.1, 0.1], [0, 0.12])
    plt.tight_layout()
        # Plot standoff range as translucent green planes
    z_min, z_max = gripper.standoff_range
    x_plane = np.linspace(-0.05, 0.05, 10)
    y_plane = np.linspace(-0.05, 0.05, 10)
    X, Y = np.meshgrid(x_plane, y_plane)

    base = gripper.get_base_mesh()
    ax.plot_trisurf(
        base.vertices[:, 0], base.vertices[:, 1], base.vertices[:, 2],
        triangles=base.faces,
        color='gray', alpha=0.3, linewidth=0.1, edgecolor='k'
    )


    for z in [z_min, z_max]:
        Z = np.full_like(X, z)
        ax.plot_surface(X, Y, Z, color='green', alpha=0.2, edgecolor='none')

    ax.text(0, 0, z_min, 'Standoff min', color='green')
    ax.text(0, 0, z_max, 'Standoff max', color='green')

    # camera at x and z cross-section
    # ax.view_init(elev=0, azim=-90)
    # plt.tight_layout()

    origins, directions = gripper.get_closing_rays()

    
    # --- Visualize valid contact range along rays ---
    arrow_origins, arrow_dirs = gripper.get_closing_rays()
    max_dist = 2.0 * gripper.q  # maximum valid distance from origin

    # Compute segment points for each ray (min and max valid hit range)
    segment_start = arrow_origins
    segment_end = arrow_origins + arrow_dirs * max_dist

    # Plot valid range as green line segments
    for start, end in zip(segment_start, segment_end):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color='limegreen', linewidth=2, alpha=0.6
        )

    # Optionally annotate one of them
    ax.text(
        segment_end[0][0], segment_end[0][1], segment_end[0][2],
        f"2q = {max_dist:.3f}m", color='green'
    )

    plt.show()


if __name__ == '__main__':
    gripper = PandaGripper(root_folder='/home/lambda1/Documents/thor-grasp/')  # Adjust if needed
    plot_gripper(gripper)
