import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import trimesh.transformations as tra
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class PandaGripper(object):
    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder=''):
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.04

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        fn_base = root_folder + 'assets/gripper_models/panda_gripper/hand.stl'
        fn_finger = root_folder + 'assets/gripper_models/panda_gripper/finger.stl'
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.ray_origins = []
        self.ray_directions = []
        for i in np.linspace(-0.01, 0.02, num_contact_points_per_finger):
            self.ray_origins.append(
                np.r_[self.finger_l.bounding_box.centroid + [0, 0, i], 1])
            self.ray_origins.append(
                np.r_[self.finger_r.bounding_box.centroid + [0, 0, i], 1])
            self.ray_directions.append(
                np.r_[-self.finger_l.bounding_box.primitive.transform[:3, 0]])
            self.ray_directions.append(
                np.r_[+self.finger_r.bounding_box.primitive.transform[:3, 0]])

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

        self.standoff_range = np.array([max(self.finger_l.bounding_box.bounds[0, 2],
                                            self.base.bounding_box.bounds[1, 2]),
                                        self.finger_l.bounding_box.bounds[1, 2]])
        self.standoff_range[0] += 0.001

    def get_meshes(self):
        return [self.finger_l, self.finger_r, self.base]

    def get_closing_rays(self, transform):
        return transform[:3, :].dot(self.ray_origins.T).T, transform[:3, :3].dot(self.ray_directions.T).T

def show_gripper_and_rays(gripper):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.clear()

    # Add the gripper mesh to the plot
    gripper.hand.show(ax=ax)

    # Plot rays using Line3DCollection
    ray_lines = []
    for origin, direction in zip(gripper.ray_origins, gripper.ray_directions):
        line_start = origin[:3]
        line_end = origin[:3] + direction * 0.1  # Increase the ray length to make it visible
        ray_lines.append([line_start, line_end])

    # Create Line3DCollection for the rays
    ray_collection = Line3DCollection(ray_lines, colors='r', linewidths=2)
    ax.add_collection(ray_collection)

    # Adjust the plot limits to ensure everything is visible
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0, 0.2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Raycasting and Gripper Visualization')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    gripper = PandaGripper(q=0.04, num_contact_points_per_finger=10)
    show_gripper_and_rays(gripper)
