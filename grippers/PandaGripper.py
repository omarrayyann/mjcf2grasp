import numpy as np
import trimesh
import trimesh.transformations as tra


class PandaGripper(object):
    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder=""):
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.04

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        fn_base = root_folder + "assets/gripper_models/panda_gripper/hand.stl"
        fn_finger = root_folder + "assets/gripper_models/panda_gripper/finger.stl"
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()
        self.tcp_offset = np.array([0, 0, 0.089275])

        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.ray_origins = []
        self.ray_directions = []
        for i in np.linspace(-0.01, 0.02, num_contact_points_per_finger):
            self.ray_origins.append(
                np.r_[self.finger_l.bounding_box.centroid + [0, 0, i], 1]
            )
            self.ray_origins.append(
                np.r_[self.finger_r.bounding_box.centroid + [0, 0, i], 1]
            )
            self.ray_directions.append(
                np.r_[-self.finger_l.bounding_box.primitive.transform[:3, 0]]
            )
            self.ray_directions.append(
                np.r_[+self.finger_r.bounding_box.primitive.transform[:3, 0]]
            )

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

        self.standoff_range = np.array(
            [
                max(
                    self.finger_l.bounding_box.bounds[0, 2],
                    self.base.bounding_box.bounds[1, 2],
                ),
                self.finger_l.bounding_box.bounds[1, 2],
            ]
        )
        self.standoff_range[0] += 0.001

    def get_obbs(self):
        return [
            self.finger_l.bounding_box,
            self.finger_r.bounding_box,
            self.base.bounding_box,
        ]

    def get_meshes(self):
        return [self.finger_l, self.finger_r, self.base]

    def get_closing_rays(self, transform):
        return transform[:3, :].dot(self.ray_origins.T).T, transform[:3, :3].dot(
            self.ray_directions.T
        ).T

    def get_finger_meshes(self):
        return [self.finger_l, self.finger_r]

    def get_base_mesh(self):
        return self.base
