import time
import numpy as np
import mujoco
import mujoco.viewer
import os
import xml.etree.ElementTree as ET
import argparse
import json
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grippers.PandaGripper import PandaGripper
from grippers.RUMGripper import RUMGripper
from grippers.RobotiqGripper import RobotiqGripper


parser = argparse.ArgumentParser()
parser.add_argument("--object_name", type=str)
parser.add_argument("--grasps_path", type=str)
parser.add_argument("--xml_file", type=str)
parser.add_argument("--gripper", type=str, default="rum")
parser.add_argument("--num_shakes", type=int, default=2)
parser.add_argument("--shake_magnitude", type=float, default=0.1)
parser.add_argument("--shake_steps", type=int, default=1000)
parser.add_argument(
    "--approach_distance",
    type=float,
    default=0.1,
    help="Distance in meters for the gripper to approach from before grasping",
)
parser.add_argument(
    "--approach_steps",
    type=int,
    default=1000,
    help="Number of simulation steps for the approach phase",
)
parser.add_argument("--render", action="store_true", help="Enable interactive viewer")
parser.add_argument("--rotate", action="store_true", help="Enable rotation shaking")
parser.add_argument(
    "--num_workers",
    type=int,
    default=max(1, mp.cpu_count() // 8),
    help="Number of parallel processes to use",
)
parser.add_argument(
    "--max_successful",
    type=int,
    default=0,
    help="Stop after finding this many successful grasps (0 = process all grasps)",
)
args = parser.parse_args()

initial_relative_position = None
initial_grasp_verified = False

if args.render:
    args.num_workers = 1


def is_object_grasped(model, data, object_name):
    left_finger_contact = False
    right_finger_contact = False

    left_patterns = ["left_finger", "finger_l", "gripper_finger_left", "left"]
    right_patterns = ["right_finger", "finger_r", "gripper_finger_right", "right"]

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        if not geom1 or not geom2:
            continue

        if object_name.lower() in geom1.lower() or object_name.lower() in geom2.lower():
            other = geom2 if object_name.lower() in geom1.lower() else geom1
            if any(p in other.lower() for p in left_patterns):
                left_finger_contact = True
            if any(p in other.lower() for p in right_patterns):
                right_finger_contact = True

    grasped = left_finger_contact and right_finger_contact

    return grasped


def check_grasp(model, data, object_name, store_initial=False):
    global initial_relative_position, initial_grasp_verified

    object_pos = data.body(object_name).xpos
    gripper_pos = data.body("base").xpos
    relative_position = object_pos - gripper_pos

    if store_initial:
        if is_object_grasped(model, data, object_name):
            initial_relative_position = relative_position.copy()
            initial_grasp_verified = True
            return True
        initial_grasp_verified = False
        return False

    if not initial_grasp_verified or initial_relative_position is None:
        return False

    position_change = np.linalg.norm(relative_position - initial_relative_position)

    grasping = is_object_grasped(model, data, object_name)
    return grasping


def test_single_grasp(grasp_data, object_name, base_xml_content, gripper_xml_content):
    i, transform, quality, config = grasp_data

    # Use pre-loaded XML content instead of file I/O in worker
    xml_content = base_xml_content

    # Use pre-loaded gripper XML content
    xml_content = merge_xml_contents(xml_content, gripper_xml_content)

    # Create combined XML for the scene with the gripper and object - EXACTLY like viewer version
    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()

    pos = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

    approach_distance = config.get("approach_distance", 0.1)
    approach_vector = transform[:3, 2] * approach_distance
    approach_pos = pos - approach_vector

    # Find the body element with name="base" and modify its position
    for body in root.findall(".//body"):
        if body.get("name") == "base":
            org_rot = R.from_matrix(transform[:3, :3])
            rot_new = org_rot  # * R.from_euler("x", 90, degrees=True)
            new_quat = rot_new.as_quat(scalar_first=True)
            body.set("pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}")
            body.set("quat", f"{new_quat[0]} {new_quat[1]} {new_quat[2]} {new_quat[3]}")
            break

    for body in root.findall(".//body"):
        if body.get("name") == "target_ee_pose":
            org_rot = R.from_matrix(transform[:3, :3])
            rot_new = org_rot  # * R.from_euler("x", 90, degrees=True)
            new_quat = rot_new.as_quat(scalar_first=True)
            body.set("pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}")
            body.set("quat", f"{new_quat[0]} {new_quat[1]} {new_quat[2]} {new_quat[3]}")
            break

    for body in root.findall(".//geom"):
        if body.get("name") == "test_sphere":
            # Get current position or set default
            body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
            body.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
            break

    # Convert the modified tree back to XML string
    xml_content = ET.tostring(root, encoding="unicode")

    # Create model and data from the XML
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    global initial_relative_position, initial_grasp_verified
    initial_relative_position = None
    initial_grasp_verified = False

    if args.gripper == "rum":
        data.ctrl[0] = 1.0
    elif args.gripper == "panda":
        data.ctrl[0] = 255.0
    elif args.gripper == "robotiq":
        data.ctrl[0] = 0.0

    mujoco.mj_step(model, data, nstep=500)

    pos = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_ee_pose")

    # Now approach by moving mocap towards target position
    approach_steps = config.get("approach_steps", 1000)
    for step in range(approach_steps):
        alpha = step / approach_steps
        current_target_pos = approach_pos + alpha * (pos - approach_pos)

        if mocap_id >= 0:
            data.mocap_pos[0] = current_target_pos

        mujoco.mj_step(model, data)

        # Check if we've reached the target
        # current_gripper_pos = data.body("base").xpos
        # distance_to_target = np.linalg.norm(current_gripper_pos - pos)
        # if distance_to_target < 0.001:
        #     break

    # Final positioning and preparation for grasping
    if mocap_id >= 0:
        data.mocap_pos[0] = pos

    # Let system stabilize
    for step in range(500):
        mujoco.mj_step(model, data)

    if args.gripper == "rum":
        data.ctrl[0] = -0.8
    elif args.gripper == "panda":
        data.ctrl[0] = 0.0
    elif args.gripper == "robotiq":
        data.ctrl[0] = 255.0

    for step in range(500):
        mujoco.mj_step(model, data)

    object_pose = np.eye(4)
    object_pose[:3, :3] = data.body(object_name).xmat.reshape(3, 3)
    object_pose[:3, 3] = data.body(object_name).xpos

    transform = np.linalg.inv(object_pose) @ transform

    if not check_grasp(model, data, object_name, store_initial=True):
        return i, None, None

    directions = ["x", "y", "z"]
    shake_success = True

    # Get the current mocap position as baseline for shaking
    if mocap_id >= 0:
        baseline_pos = data.mocap_pos[0].copy()
        baseline_quat = data.mocap_quat[0].copy()
    else:
        baseline_pos = pos.copy()
        baseline_quat = quat.copy()

    for direction_idx, direction in enumerate(directions):
        for shake in range(config["num_shakes"]):
            total_steps = config["shake_steps"] * 2
            if args.gripper == "panda":
                total_steps = total_steps * 4

            for step in range(total_steps):
                angle = 2 * np.pi * step / total_steps
                shake_offset = config["shake_magnitude"] * np.sin(angle)

                # Create shake position by adding offset in the specified direction
                shake_pos = baseline_pos.copy()
                shake_pos[direction_idx] += shake_offset

                # Apply shake via mocap
                if mocap_id >= 0:
                    data.mocap_pos[0] = shake_pos

                mujoco.mj_step(model, data)

                if step == total_steps // 4 or step == 3 * total_steps // 4:
                    grasp_maintained = check_grasp(model, data, object_name)
                    if not grasp_maintained:
                        shake_success = False
                        break

            if not shake_success:
                break

            # Return to baseline position
            if mocap_id >= 0:
                data.mocap_pos[0] = baseline_pos
            for step in range(50):
                mujoco.mj_step(model, data)

        if not shake_success:
            break

    if args.rotate and shake_success:
        if shake_success and mocap_id >= 0:
            # Convert baseline quaternion to rotation matrix for manipulation
            baseline_rotation = R.from_quat(
                baseline_quat[[1, 2, 3, 0]]
            )  # MuJoCo uses w,x,y,z, scipy uses x,y,z,w

            # Test rotations around x (roll), y (pitch), z (yaw) axes
            for axis_idx in range(2, 3):  # x, y, z axes
                for shake in range(1):
                    total_steps = config["shake_steps"] * 5
                    if args.gripper == "panda":
                        total_steps = total_steps * 2

                    for step in range(total_steps):
                        angle = 2 * np.pi * step / total_steps
                        shake_angle = angle  # Full 360 degree rotation

                        # Create rotation around the axis
                        rotation_vec = np.zeros(3)
                        rotation_vec[axis_idx] = shake_angle
                        shake_rotation = R.from_rotvec(rotation_vec)

                        # Apply rotation to baseline orientation
                        new_rotation = baseline_rotation * shake_rotation
                        new_quat_scipy = new_rotation.as_quat()  # x,y,z,w format
                        new_quat_mujoco = np.array(
                            [
                                new_quat_scipy[3],
                                new_quat_scipy[0],
                                new_quat_scipy[1],
                                new_quat_scipy[2],
                            ]
                        )  # w,x,y,z format

                        # Apply rotational displacement
                        data.mocap_quat[0] = new_quat_mujoco

                        mujoco.mj_step(model, data)

                        # Check grasp integrity at quarter and three-quarter points
                        if step == total_steps // 4 or step == 3 * total_steps // 4:
                            grasp_maintained = check_grasp(model, data, object_name)
                            if not grasp_maintained:
                                shake_success = False
                                break

                    if not shake_success:
                        break

                    # Return to baseline orientation
                    data.mocap_quat[0] = baseline_quat
                    for step in range(50):
                        mujoco.mj_step(model, data)

                if not shake_success:
                    break

    final_grasp_check = is_object_grasped(model, data, object_name)

    if shake_success and final_grasp_check:
        return i, transform.tolist(), quality
    else:
        return i, None, None


def run_simulation_with_viewer(xml_content, object_name, use_viewer):
    if use_viewer:
        with open(args.grasps_path, "r") as f:
            grasp_data = json.load(f)
        transforms = np.array(grasp_data["transforms"])
        qualities = np.array(
            grasp_data.get("quality_antipodal", [1.0] * len(transforms))
        )
        widths = np.array(grasp_data.get("grasp_widths", [0.05] * len(transforms)))

        successful_transforms = []
        successful_qualities = []
        successful_widths = []

        pbar = tqdm(
            enumerate(zip(transforms, qualities)),
            total=len(transforms),
            desc="Testing grasps (0/0 successful)",
        )

        viewer = None

        gripper = None
        if args.gripper == "panda":
            gripper = PandaGripper()
        elif args.gripper == "rum":
            gripper = RUMGripper()
        elif args.gripper == "robotiq":
            gripper = RobotiqGripper()

        for i, (transform, quality) in pbar:
            transform = np.array(
                [
                    [
                        -9.77128557e-01,
                        -7.02050876e-03,
                        -2.12533516e-01,
                        -1.95007307e-03,
                    ],
                    [2.12649228e-01, -3.08611412e-02, -9.76641129e-01, -1.11673942e-02],
                    [2.97490752e-04, -9.99499026e-01, 3.16482082e-02, 2.46439797e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            )
            tree = ET.ElementTree(ET.fromstring(xml_content))
            root = tree.getroot()

            pos = transform[:3, 3]
            quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

            approach_distance = args.approach_distance
            approach_vector = transform[:3, 2] * approach_distance
            approach_pos = pos - approach_vector

            for body in root.findall(".//body"):
                if body.get("name") == "base":
                    org_rot = R.from_matrix(transform[:3, :3])
                    rot_new = org_rot  # * R.from_euler("x", 90, degrees=True)
                    new_quat = rot_new.as_quat(scalar_first=True)
                    body.set(
                        "pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}"
                    )
                    body.set(
                        "quat",
                        f"{new_quat[0]} {new_quat[1]} {new_quat[2]} {new_quat[3]}",
                    )
                    break

            for body in root.findall(".//body"):
                if body.get("name") == "target_ee_pose":
                    org_rot = R.from_matrix(transform[:3, :3])
                    rot_new = org_rot  #  * R.from_euler("x", 90, degrees=True)
                    new_quat = rot_new.as_quat(scalar_first=True)
                    body.set(
                        "pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}"
                    )
                    body.set(
                        "quat",
                        f"{new_quat[0]} {new_quat[1]} {new_quat[2]} {new_quat[3]}",
                    )
                    break

            for body in root.findall(".//geom"):
                if body.get("name") == "test_sphere":
                    # Get current position or set default
                    body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
                    body.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
                    break

            # Convert the modified tree back to XML string
            xml_content = ET.tostring(root, encoding="unicode")

            # Create model and data from the XML
            model = mujoco.MjModel.from_xml_string(xml_content)
            data = mujoco.MjData(model)

            if viewer is not None:
                viewer.close()

            time.sleep(0.1)
            with mujoco.viewer.launch_passive(
                model, data, show_left_ui=True, show_right_ui=True
            ) as viewer:
                mujoco.mj_step(model, data)
                viewer.sync()

                if args.gripper == "panda":
                    data.ctrl[0] = 255.0
                elif args.gripper == "rum":
                    data.ctrl[0] = 1.0
                elif args.gripper == "robotiq":
                    data.ctrl[0] = 0.0

                mujoco.mj_step(model, data, nstep=2000)

                pos = transform[:3, 3]
                quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

                mocap_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, "target_ee_pose"
                )

                # Now approach by moving mocap towards target position
                approach_steps = args.approach_steps
                for step in range(approach_steps):
                    alpha = step / approach_steps
                    current_target_pos = approach_pos + alpha * (pos - approach_pos)

                    if mocap_id >= 0:
                        data.mocap_pos[0] = current_target_pos

                    mujoco.mj_step(model, data)

                    if step % 50 == 0:
                        viewer.sync()

                    # Check if we've reached the target
                    # current_gripper_pos = data.body("base").xpos
                    # distance_to_target = np.linalg.norm(current_gripper_pos - pos)

                # Final positioning and preparation for grasping
                if mocap_id >= 0:
                    data.mocap_pos[0] = pos

                # while 1:
                #     mujoco.mj_step(model, data)
                #     viewer.sync()

                # Let system stabilize
                for step in range(100):
                    mujoco.mj_step(model, data)
                    if step % 20 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            viewer.close()
                            return ([], [], [])

                for step in range(4000):
                    mujoco.mj_step(model, data)
                    if step % 50 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            viewer.close()
                            return (
                                successful_transforms,
                                successful_qualities,
                                successful_widths,
                            )

                if args.gripper == "rum":
                    data.ctrl[0] = -0.8
                elif args.gripper == "panda":
                    data.ctrl[0] = 0.0
                elif args.gripper == "robotiq":
                    data.ctrl[0] = 255.0

                for step in range(4000):
                    mujoco.mj_step(model, data)
                    if step % 20 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            viewer.close()
                            return (
                                successful_transforms,
                                successful_qualities,
                                successful_widths,
                            )

                tcp_pose = np.eye(4)
                tcp_pose[:3, 3] = data.site("tcp").xpos
                tcp_pose[:3, :3] = data.site("tcp").xmat.reshape(3, 3)

                # rot_x_90 = R.from_euler("x", -90, degrees=True).as_matrix()
                # rotated_mat = mat @ rot_x_90

                object_pose = np.eye(4)
                object_pose[:3, :3] = data.body(object_name).xmat.reshape(3, 3)
                object_pose[:3, 3] = data.body(object_name).xpos

                print(f"Before: {transform}")
                transform = np.linalg.inv(object_pose) @ tcp_pose

                print(f"After: {transform}")

                if not check_grasp(model, data, object_name, store_initial=True):
                    pbar.set_description(
                        f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                    )
                    viewer.close()
                    continue

                directions = ["x", "y", "z"]
                shake_success = True

                # Get the current mocap position as baseline for shaking
                if mocap_id >= 0:
                    baseline_pos = data.mocap_pos[0].copy()
                    baseline_quat = data.mocap_quat[0].copy()
                else:
                    baseline_pos = pos.copy()
                    baseline_quat = quat.copy()

                for direction_idx, direction in enumerate(directions):
                    for shake in range(args.num_shakes):
                        total_steps = args.shake_steps * 2
                        if args.gripper == "panda":
                            total_steps = total_steps * 4

                        for step in range(total_steps):
                            angle = 2 * np.pi * step / total_steps
                            shake_offset = args.shake_magnitude * np.sin(angle)

                            # Create shake position by adding offset in the specified direction
                            shake_pos = baseline_pos.copy()
                            shake_pos[direction_idx] += shake_offset

                            # Apply shake via mocap
                            if mocap_id >= 0:
                                data.mocap_pos[0] = shake_pos

                            mujoco.mj_step(model, data)
                            if step % 5 == 0:
                                viewer.sync()
                                if not viewer.is_running():
                                    viewer.close()
                                    return (
                                        successful_transforms,
                                        successful_qualities,
                                        successful_widths,
                                    )

                            if step == total_steps // 4 or step == 3 * total_steps // 4:
                                grasp_maintained = check_grasp(model, data, object_name)
                                if not grasp_maintained:
                                    shake_success = False
                                    viewer.close()
                                    break

                        if not shake_success:
                            viewer.close()
                            break

                        # Return to baseline position
                        if mocap_id >= 0:
                            data.mocap_pos[0] = baseline_pos
                        for step in range(50):
                            mujoco.mj_step(model, data)
                            if step % 20 == 0:
                                viewer.sync()
                                if not viewer.is_running():
                                    viewer.close()
                                    return (
                                        successful_transforms,
                                        successful_qualities,
                                        successful_widths,
                                    )

                    if not shake_success:
                        viewer.close()
                        break

                if args.rotate and shake_success:
                    # Test rotational shaking in roll, pitch, yaw after translational shaking
                    if shake_success and mocap_id >= 0:
                        # Convert baseline quaternion to rotation matrix for manipulation
                        baseline_rotation = R.from_quat(
                            baseline_quat[[1, 2, 3, 0]]
                        )  # MuJoCo uses w,x,y,z, scipy uses x,y,z,w

                        # Test rotations around x (roll), y (pitch), z (yaw) axes
                        for axis_idx in range(2, 3):  # x, y, z axes
                            for shake in range(1):
                                total_steps = args.shake_steps * 5
                                if args.gripper == "panda":
                                    total_steps = total_steps * 2

                                for step in range(total_steps):
                                    angle = 2 * np.pi * step / total_steps
                                    shake_angle = angle  # Full 360 degree rotation

                                    # Create rotation around the axis
                                    rotation_vec = np.zeros(3)
                                    rotation_vec[axis_idx] = shake_angle
                                    shake_rotation = R.from_rotvec(rotation_vec)

                                    # Apply rotation to baseline orientation
                                    new_rotation = baseline_rotation * shake_rotation
                                    new_quat_scipy = (
                                        new_rotation.as_quat()
                                    )  # x,y,z,w format
                                    new_quat_mujoco = np.array(
                                        [
                                            new_quat_scipy[3],
                                            new_quat_scipy[0],
                                            new_quat_scipy[1],
                                            new_quat_scipy[2],
                                        ]
                                    )  # w,x,y,z format

                                    # Apply rotational displacement
                                    data.mocap_quat[0] = new_quat_mujoco

                                    mujoco.mj_step(model, data)
                                    if step % 5 == 0:
                                        viewer.sync()
                                        if not viewer.is_running():
                                            return (
                                                successful_transforms,
                                                successful_qualities,
                                                successful_widths,
                                            )

                                    # Check grasp integrity at quarter and three-quarter points
                                    if (
                                        step == total_steps // 4
                                        or step == 3 * total_steps // 4
                                    ):
                                        grasp_maintained = check_grasp(
                                            model, data, object_name
                                        )
                                        if not grasp_maintained:
                                            shake_success = False
                                            viewer.close()
                                            break

                                if not shake_success:
                                    viewer.close()
                                    break

                                # Return to baseline orientation
                                data.mocap_quat[0] = baseline_quat
                                for step in range(50):
                                    mujoco.mj_step(model, data)
                                    if step % 20 == 0:
                                        viewer.sync()
                                        if not viewer.is_running():
                                            viewer.close()
                                            return (
                                                successful_transforms,
                                                successful_qualities,
                                                successful_widths,
                                            )

                            if not shake_success:
                                viewer.close()
                                break

                final_grasp_check = is_object_grasped(model, data, object_name)

                if shake_success and final_grasp_check:
                    successful_transforms.append(transform.tolist())
                    successful_qualities.append(quality)
                    successful_widths.append(widths[i])

                    if (
                        args.max_successful > 0
                        and len(successful_transforms) >= args.max_successful
                    ):
                        tqdm.write(
                            f"Found {len(successful_transforms)} successful grasps (reached max_successful limit)"
                        )
                        viewer.close()
                        return (
                            successful_transforms,
                            successful_qualities,
                            successful_widths,
                        )

                pbar.set_description(
                    f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                )

                # for _ in range(100):
                #     mujoco.mj_step(model, data)
                #     viewer.sync()
                #     if not viewer.is_running():
                #         return (
                #             successful_transforms,
                #             successful_qualities,
                #             successful_widths,
                #         )

        return successful_transforms, successful_qualities, successful_widths

    else:
        with open(args.grasps_path, "r") as f:
            grasp_data = json.load(f)
        transforms = np.array(grasp_data["transforms"])
        qualities = np.array(
            grasp_data.get("quality_antipodal", [1.0] * len(transforms))
        )
        widths = np.array(grasp_data.get("grasp_widths", [0.05] * len(transforms)))

        config = {
            "num_shakes": args.num_shakes,
            "shake_magnitude": args.shake_magnitude,
            "shake_steps": args.shake_steps,
            "approach_distance": args.approach_distance,
            "approach_steps": args.approach_steps,
        }

        # Pre-load XML content to avoid file I/O in workers
        xml_path = os.path.join(os.path.dirname(__file__), "../assets/scene.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        include = ET.Element("include", {"file": args.xml_file})
        root.append(include)
        base_xml_content = ET.tostring(root, encoding="unicode")
        
        gripper_xml_path = os.path.join(
            os.path.dirname(__file__),
            f"../assets/gripper_models/{args.gripper}_gripper/model.xml",
        )
        with open(gripper_xml_path, "r") as f:
            gripper_xml_content = f.read()

        grasp_params = [
            (i, transform, quality, config)
            for i, (transform, quality) in enumerate(zip(transforms, qualities))
        ]

        num_workers = min(args.num_workers, len(grasp_params))

        successful_transforms = []
        successful_qualities = []
        successful_widths = []

        with mp.Manager() as manager:
            success_count = manager.Value("i", 0)
            processed_count = manager.Value("i", 0)
            lock = manager.Lock()

            should_stop = manager.Value("b", False)

            def update_progress_bar(result):
                nonlocal pbar
                i, transform_result, quality_result = result

                with lock:
                    processed_count.value += 1

                    if transform_result is not None:
                        success_count.value += 1
                        successful_transforms.append(
                            (i, transform_result, quality_result, widths[i])
                        )

                        if (
                            args.max_successful > 0
                            and success_count.value >= args.max_successful
                        ):
                            should_stop.value = True
                            tqdm.write(
                                f"Found {success_count.value} successful grasps (reached max_successful limit)"
                            )

                pbar.set_description(
                    f"Testing grasps ({success_count.value}/{processed_count.value} successful)"
                )
                pbar.update(1)

            pbar = tqdm(total=len(grasp_params), desc="Testing grasps (0/0 successful)")

            with mp.Pool(processes=num_workers) as pool:
                results = [
                    pool.apply_async(
                        test_single_grasp,
                        args=(param, object_name, base_xml_content, gripper_xml_content),
                        callback=update_progress_bar,
                    )
                    for param in grasp_params
                ]

                completed = 0
                timeout_counter = 0
                max_timeout = 300  # 5 minutes timeout
                
                while completed < len(results):
                    if should_stop.value:
                        # Graceful shutdown instead of abrupt termination
                        pool.close()
                        pool.join()
                        tqdm.write(
                            "Stopped processing after reaching max successful grasps"
                        )
                        break

                    # Check for timeout to prevent infinite waiting
                    if timeout_counter > max_timeout:
                        tqdm.write("Timeout reached, terminating pool")
                        pool.terminate()
                        pool.join()
                        break
                    
                    timeout_counter += 1
                    
                    for i, r in enumerate(results):
                        if r is not None and r.ready() and not r.successful():
                            try:
                                r.get(timeout=1)  # Add timeout to get()
                            except Exception as e:
                                tqdm.write(f"Worker error: {str(e)}")
                            results[i] = None
                            completed += 1
                        elif r is not None and r.ready():
                            results[i] = None
                            completed += 1

                if not should_stop.value:
                    pool.close()
                    pool.join()

            successful_transforms.sort()

            successful_transforms_only = [t for _, t, _, _ in successful_transforms]
            successful_qualities_only = [q for _, _, q, _ in successful_transforms]
            successful_widths_only = [w for _, _, _, w in successful_transforms]

        pbar.close()
        return (
            successful_transforms_only,
            successful_qualities_only,
            successful_widths_only,
        )


def merge_xml_contents(base_xml_content, additional_xml_content):
    base_root = ET.fromstring(base_xml_content)
    additional_root = ET.fromstring(additional_xml_content)

    if base_root.tag != "mujoco" or additional_root.tag != "mujoco":
        raise ValueError("Both XML contents must have 'mujoco' as the root element")

    processed_sections = {}

    for additional_child in additional_root:
        tag_name = additional_child.tag

        base_section = base_root.find(tag_name)

        if base_section is not None:
            if tag_name in processed_sections:
                continue

            for element in additional_child:
                is_duplicate = False
                for existing in base_section:
                    if element.tag == existing.tag and all(
                        attr in existing.attrib and existing.attrib[attr] == val
                        for attr, val in element.attrib.items()
                        if attr != "name"
                    ):
                        if "name" in element.attrib and "name" in existing.attrib:
                            if element.attrib["name"] == existing.attrib["name"]:
                                is_duplicate = True
                                break
                        else:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    base_section.append(element)

            processed_sections[tag_name] = True
        else:
            base_root.append(additional_child)
            processed_sections[tag_name] = True

    return ET.tostring(base_root, encoding="unicode")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    xml_path = os.path.join(os.path.dirname(__file__), "../assets/scene.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_name = args.object_name

    include = ET.Element("include", {"file": args.xml_file})
    root.append(include)
    xml_content = ET.tostring(root, encoding="unicode")

    gripper_xml_path = os.path.join(
        os.path.dirname(__file__),
        f"../assets/gripper_models/{args.gripper}_gripper/model.xml",
    )
    with open(gripper_xml_path, "r") as f:
        additional_xml_content = f.read()
    xml_content = merge_xml_contents(xml_content, additional_xml_content)

    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    successful_transforms, successful_qualities, successful_widths = (
        run_simulation_with_viewer(xml_content, object_name, args.render)
    )

    output_path = args.grasps_path.replace(".json", "_filtered.json")
    with open(output_path, "w") as f:
        with open(args.grasps_path, "r") as original_f:
            original_data = json.load(original_f)

        json.dump(
            {
                "transforms": successful_transforms,
                "quality_antipodal": successful_qualities,
                "object": original_data.get("object", "unknown_object"),
                "object_scale": original_data.get("object_scale", 1.0),
                "object_position": original_data.get("object_position", [0, 0, 0]),
                "object_rotation": original_data.get("object_rotation", [1, 0, 0, 0]),
                "approach_distance": args.approach_distance,
                "grasp_widths": successful_widths,
            },
            f,
            indent=2,
        )

    tqdm.write(f"Saved {len(successful_transforms)} successful grasps to {output_path}")

    if not args.render and args.num_workers > 1:
        tqdm.write(f"Used {args.num_workers} parallel workers for grasp testing")