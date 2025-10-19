import numpy as np
import trimesh
import trimesh.transformations as tra


class RobotiqGripper:
    tcp_offset = np.array([0, 0, 0.13675])

    def __init__(self, q=0.048372, num_contact_points_per_finger=15, root_folder=""):
        self.default_pregrasp_configuration = 0.048372

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        gripping_center = 0.17365
        self.tcp_offset = np.array([0, 0, gripping_center])
        fn_base = root_folder + "assets/grippers/robotiq/xmls/hand.stl"
        fn_finger = root_folder + "assets/grippers/robotiq/xmls/finger.stl"
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.13686])
        self.finger_r.apply_translation([-q, 0, 0.13686])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.ray_origins = []
        self.ray_directions = []
        for i in np.linspace(0.001, 0.036, num_contact_points_per_finger):
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
                0.135342,
                self.tcp_offset[2],
            ]
        )
        self.standoff_range[0] += 0.001

    def get_base_mesh(self):
        return self.base

    def get_base_obb(self):
        return self.base.bounding_box

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
