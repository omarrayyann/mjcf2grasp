import matplotlib
matplotlib.use('Agg')
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import trimesh.transformations as tra
import time

class RumGripper(object):
    def __init__(self, angle_deg=None, num_contact_points_per_finger=10):
        self.default_pregrasp_configuration = 0
        if angle_deg is None:
            angle_deg = self.default_pregrasp_configuration

        self.mount_offset_r = np.array([0, 0, 0.0])
        self.pivot_r = np.array([-0.023, 0, 0.04224])

        self.mount_offset_l = np.array([0, 0, 0.0])
        self.pivot_l = np.array([0.023, 0, 0.04224])

        self.angle_rad = np.deg2rad(angle_deg)
        base_path = '/home/lambda1/Documents/thor-grasp/assets/rum_gripper/meshes/'
        self.body = trimesh.load(base_path + 'new_body(1).stl')
        self.finger_l = trimesh.load(base_path + 'new_left(1).stl')
        self.finger_r = trimesh.load(base_path + 'new_right(1).stl')

        self.reset_fingers()

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.body, self.finger_l, self.finger_r])

        self.ray_origins = []
        self.ray_directions = []

        

        for i in np.linspace(-0.01, 0.02, num_contact_points_per_finger):
            origin_l = self.finger_l.bounding_box.centroid + np.array([0, 0, i])
            origin_r = self.finger_r.bounding_box.centroid + np.array([0, 0, i])
            dir_l = (self.pivot_r - origin_l)
            dir_r = (self.pivot_l - origin_r)
            dir_l /= np.linalg.norm(dir_l)
            dir_r /= np.linalg.norm(dir_r)

            self.ray_origins.append(np.r_[origin_l, 1])
            self.ray_directions.append(dir_l)
            self.ray_origins.append(np.r_[origin_r, 1])
            self.ray_directions.append(dir_r)

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

        self.q = np.deg2rad(angle_deg)
        

    def reset_fingers(self):
        self.finger_l = trimesh.load('/home/lambda1/Documents/thor-grasp/assets/rum_gripper/meshes/new_left(1).stl')
        self.finger_r = trimesh.load('/home/lambda1/Documents/thor-grasp/assets/rum_gripper/meshes/new_right(1).stl')

        
        rot_l = tra.rotation_matrix(self.angle_rad, [0, -1, 0], point=self.pivot_l)
        self.finger_l.apply_transform(rot_l)
        self.finger_l.apply_translation(self.mount_offset_l)

        
        rot_r = tra.rotation_matrix(-self.angle_rad, [0, -1, 0], point=self.pivot_r)
        self.finger_r.apply_transform(rot_r)
        self.finger_r.apply_translation(self.mount_offset_r)

    def update_finger_position(self, angle_deg):
        self.reset_fingers()

        angle_rad = np.deg2rad(angle_deg)

        rot_l = tra.rotation_matrix(angle_rad, [0, -1, 0], point=self.pivot_l)
        self.finger_l.apply_transform(rot_l)

        rot_r = tra.rotation_matrix(-angle_rad, [0, -1, 0], point=self.pivot_r)
        self.finger_r.apply_transform(rot_r)

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.body, self.finger_l, self.finger_r])


gripper = RumGripper(angle_deg=0, num_contact_points_per_finger=10)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

def show_gripper():
    ax.clear()

    gripper.hand.show(ax=ax)

    for origin, direction in zip(gripper.ray_origins, gripper.ray_directions):
        ax.quiver(origin[0], origin[1], origin[2], direction[0], direction[1], direction[2], length=0.05, color='r', arrow_length_ratio=0.1)

    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0, 0.1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Raycasting Visualization for Rum Gripper')

gripper_open = False

while True:
    if gripper_open:
        gripper.update_finger_position(0)
    else:
        print(1)
        gripper.update_finger_position(45)

    show_gripper()

    gripper_open = not gripper_open

    plt.pause(1)
