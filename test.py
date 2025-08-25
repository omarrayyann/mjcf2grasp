#!/usr/bin/env python3
"""
panda_gripper_with_frame.py

Build a PandaGripper mesh (base + fingers), recenter at origin,
add coordinate frame arrows, and export to OBJ+MTL.
"""

import argparse
import numpy as np
import trimesh
import trimesh.transformations as tra
from trimesh.creation import cylinder, cone


class PandaGripperMesh:
    def __init__(self, q=None, root_folder=""):
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.04

        if q is None:
            q = self.default_pregrasp_configuration
        self.q = q

        # Load meshes
        fn_base = root_folder + "assets/gripper_models/panda_gripper/hand.stl"
        fn_finger = root_folder + "assets/gripper_models/panda_gripper/finger.stl"
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        # Place fingers relative to base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        # Concatenate
        self.hand = trimesh.util.concatenate([self.base, self.finger_l, self.finger_r])

        # Center at origin (optional: move to bounding box centroid)
        self.hand.apply_translation(-self.hand.centroid)

    def get_mesh(self):
        return self.hand


def axis_arrow(direction, length, shaft_radius, tip_length, tip_radius, name):
    """
    Make an arrow along 'direction' with cylinder + cone.
    """
    shaft = cylinder(radius=shaft_radius, height=length - tip_length, sections=32)
    shaft.apply_translation([0, 0, (length - tip_length) / 2.0])
    shaft.metadata["name"] = f"{name}_shaft"

    tip = cone(radius=tip_radius, height=tip_length, sections=32)
    tip.apply_translation([0, 0, length - tip_length / 2.0])
    tip.metadata["name"] = f"{name}_tip"

    R = trimesh.geometry.align_vectors([0, 0, 1], direction)
    shaft.apply_transform(R)
    tip.apply_transform(R)

    return [shaft, tip]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="panda_with_frame.obj")
    ap.add_argument("--root_folder", type=str, default="./")
    ap.add_argument("--axis_length", type=float, default=0.1)
    ap.add_argument("--shaft_radius", type=float, default=0.003)
    ap.add_argument("--tip_length", type=float, default=0.03)
    ap.add_argument("--tip_radius", type=float, default=0.008)
    args = ap.parse_args()

    # Build gripper
    gripper = PandaGripperMesh(root_folder=args.root_folder)
    hand_mesh = gripper.get_mesh()

    geoms = [hand_mesh]

    # Add coordinate frame
    geoms += axis_arrow(
        [1, 0, 0],
        args.axis_length,
        args.shaft_radius,
        args.tip_length,
        args.tip_radius,
        "X_axis",
    )
    geoms += axis_arrow(
        [0, 1, 0],
        args.axis_length,
        args.shaft_radius,
        args.tip_length,
        args.tip_radius,
        "Y_axis",
    )
    geoms += axis_arrow(
        [0, 0, 1],
        args.axis_length,
        args.shaft_radius,
        args.tip_length,
        args.tip_radius,
        "Z_axis",
    )

    # Export
    scene = trimesh.Scene(geoms)
    scene.export(args.output)
    print(f"✅ Exported PandaGripper with frame to {args.output}")


if __name__ == "__main__":
    main()
