import argparse
import json
import os
import numpy as np
import multiprocessing as mp
import xml.etree.ElementTree as ET
import time
import traceback
from tqdm import tqdm
import mujoco
import mujoco.viewer

from scipy.spatial.transform import Rotation as R
import re

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grippers.PandaGripper import PandaGripper
from grippers.RUMGripper import RUMGripper
from grippers.RobotiqGripper import RobotiqGripper


def rotation_matrix_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )


def rotate_vector(vector, rotation_matrix):
    return np.dot(rotation_matrix, vector)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])


def axis_angle_to_quat(axis, angle):
    axis = axis / np.linalg.norm(axis)
    half_angle = angle * 0.5
    s = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s])


def get_joint_position(model, data, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint '{joint_name}' not found in the model.")
    return data.joint(joint_id).qpos.copy()


def check_sufficient_joint_movement(
    joint_positions,
):
    max_joint_position = max(joint_positions)
    min_joint_position = min(joint_positions)
    if max_joint_position - min_joint_position < 0.01:
        return False
    return True


def is_grasping(model, data, handle_geoms):
    left_patterns = ["left_finger", "finger_l", "gripper_finger_left", "left"]
    right_patterns = ["right_finger", "finger_r", "gripper_finger_right", "right"]

    handle_geoms = [
        re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", geom) for geom in handle_geoms
    ]
    for i in range(data.ncon):
        contact = data.contact[i]

        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        if not geom1 or not geom2:
            continue

        if any(
            [
                handle_geom.lower() in geom1.lower()
                or geom1.lower() in handle_geom.lower()
                for handle_geom in handle_geoms
            ]
        ) or any(
            [
                handle_geom.lower() in geom2.lower()
                or geom2.lower() in handle_geom.lower()
                for handle_geom in handle_geoms
            ]
        ):
            other = (
                geom2
                if np.any(
                    [
                        handle_geom.lower() in geom1.lower()
                        for handle_geom in handle_geoms
                    ]
                )
                else geom1
            )
            if any(p in other.lower() for p in left_patterns) or any(
                p in other.lower() for p in right_patterns
            ):
                return True

    return False


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


def is_object_grasped(model, data, object_name):
    left_finger_contact = False
    right_finger_contact = False

    left_patterns = ["left_finger", "finger_l", "gripper_finger_left"]
    right_patterns = ["right_finger", "finger_r", "gripper_finger_right"]

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

    return left_finger_contact and right_finger_contact


def check_grasp(model, data, object_name, store_initial=False):
    global initial_relative_position, initial_grasp_verified

    object_pos = data.body(object_name).xpos
    gripper_pos = data.site("end_effector").xpos
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
    return position_change < 0.03 and is_object_grasped(model, data, object_name)


def test_single_grasp(
    grasp_data,
    object_name,
    xml_content,
    args,
    handle_geoms,
    primary_joint=None,
    render=False,
):
    joint_info = primary_joint
    if len(handle_geoms) == 0:
        print(
            f"[ERROR] No handle geometry info provided for joint '{object_name}'. Skipping this grasp."
        )
        return grasp_data[0], None, None
    if not joint_info:
        print(
            f"[ERROR] No joint axis info provided for joint '{object_name}'. Skipping this grasp."
        )
        return grasp_data[0], None, None

    i, transform, quality, config = grasp_data

    # new_transform = transform.copy()
    # rot_x = R.from_euler("x", 180, degrees=True).as_matrix()
    # new_transform[:3, :3] = new_transform[:3, :3] @ rot_x
    # transform = new_transform

    pos = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

    approach_distance = args.approach_distance
    approach_steps = args.approach_steps
    approach_vector = transform[:3, 2] * approach_distance
    approach_pos = pos - approach_vector

    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()
    gripper_base = root.find(".//body[@name='base']")
    if gripper_base is not None:
        gripper_base.set(
            "pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}"
        )
        gripper_base.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
    target_ee_pose = root.find(".//body[@name='target_ee_pose']")
    if target_ee_pose is not None:
        target_ee_pose.set(
            "pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}"
        )
        target_ee_pose.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
    if args.gripper == "rum":
        data.ctrl[0] = 1.0
    elif args.gripper == "panda":
        data.ctrl[0] = 255.0
    elif args.gripper == "robotiq":
        data.ctrl[0] = 0.0

    for _ in range(500):
        mujoco.mj_step(model, data)
        if render and viewer is not None:
            viewer.sync()

    if 1:
        model.site_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "origin")] = (
            transform[:3, 3]
        )
        model.site_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "x_axis")] = (
            transform[:3, 3] + transform[:3, 0] * 0.1
        )
        model.site_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y_axis")] = (
            transform[:3, 3] + transform[:3, 1] * 0.1
        )
        model.site_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "z_axis")] = (
            transform[:3, 3] + transform[:3, 2] * 0.1
        )

        mujoco.mj_step(model, data)
        if render and viewer is not None:
            viewer.sync()

    joint_name = None
    if joint_info and "primary_joint" in joint_info:
        joint_name = joint_info["primary_joint"]["name"]
    elif joint_info and "name" in joint_info:
        joint_name = joint_info["name"]
    if not joint_name:
        if render and viewer is not None:
            viewer.close()
            del viewer
            time.sleep(0.1)
        return i, None, None
    joint_position_before = get_joint_position(model, data, joint_name)

    for step in range(approach_steps):
        new_pos = approach_pos + (step / approach_steps) * approach_vector
        data.mocap_pos[0] = new_pos
        data.mocap_quat[0] = quat
        for i in range(100):
            mujoco.mj_step(model, data)
            if render and viewer is not None:
                viewer.sync()

    joint_position_after = get_joint_position(model, data, joint_name)

    if joint_position_before is None or joint_position_after is None:
        if render and viewer is not None:
            viewer.close()
            del viewer
            time.sleep(0.1)
        return i, None, None
    if np.abs(joint_position_after - joint_position_before) > 0.01:
        if render and viewer is not None:
            viewer.close()
            del viewer
            time.sleep(0.1)
        return i, None, None

    data.mocap_pos[0] = pos
    mujoco.mj_step(model, data, nstep=500)
    if render and viewer is not None:
        viewer.sync()

    if args.gripper == "rum":
        data.ctrl[0] = -0.8
    elif args.gripper == "panda":
        data.ctrl[0] = 0.0
    elif args.gripper == "robotiq":
        data.ctrl[0] = 255.0

    # is_grasping(model, data, handle_geoms)
    for _ in range(500):
        mujoco.mj_step(model, data)
        if render and viewer is not None:
            viewer.sync()

    while 1:
        mujoco.mj_step(model, data)
        if render and viewer is not None:
            viewer.sync()
        break

    if not is_grasping(model, data, handle_geoms):
        if render and viewer is not None:
            viewer.close()
            del viewer
            time.sleep(0.1)
        return i, None, None

    num_waypoints = 2000
    waypoints = []
    primary_joint_data = None
    if joint_info and "primary_joint" in joint_info:
        primary_joint_data = joint_info["primary_joint"]
    elif joint_info:
        primary_joint_data = joint_info
    if primary_joint_data:
        joint_type = primary_joint_data.get("type")
        joint_range_str = primary_joint_data.get("range", "0 0")
        joint_range = [float(x) for x in joint_range_str.split()]

        gripper_pos = data.site("tcp").xpos.copy()
        gripper_rot = data.site("tcp").xmat.copy().reshape(3, 3)
        gripper_quat = quat

        if joint_type == "hinge" or joint_type == "unknown":
            rotation_axis = primary_joint_data.get(
                "rotation_axis", {"x": 0, "y": 0, "z": 0}
            )
            axis_world = np.array(
                [
                    rotation_axis.get("x", 0),
                    rotation_axis.get("y", 0),
                    rotation_axis.get("z", 0),
                ]
            )
            joint_position = primary_joint_data.get(
                "position", {"x": 0, "y": 0, "z": 0}
            )
            pivot_point = np.array(
                [
                    joint_position.get("x", 0),
                    joint_position.get("y", 0),
                    joint_position.get("z", 0),
                ]
            )

            max_angle = joint_range[1]
            if max_angle == 0:
                max_angle = np.pi / 2

            rel_pos = gripper_pos - pivot_point
            for i in range(num_waypoints + 1):
                angle = i * max_angle / num_waypoints
                rotation_matrix = rotation_matrix_from_axis_angle(axis_world, angle)
                new_rel_pos = rotate_vector(rel_pos, rotation_matrix)
                new_pos = pivot_point + new_rel_pos
                rotation_quat = axis_angle_to_quat(axis_world, angle)
                new_quat = quat_multiply(rotation_quat, gripper_quat)
                waypoints.append((new_pos, new_quat))

        elif joint_type == "slide":
            axis_str = primary_joint_data.get("axis", "0 0 0")
            slide_axis = np.array([float(x) for x in axis_str.split()])
            max_distance = joint_range[1]
            if max_distance == 0:
                max_distance = 0.2
            for i in range(num_waypoints + 1):
                distance = i * max_distance / num_waypoints
                new_pos = gripper_pos + slide_axis * distance
                waypoints.append((new_pos, gripper_quat.copy()))

    joint_positions = []
    articulation_success = True
    mujoco.mj_step(model, data, nstep=2000)

    if waypoints:
        num_loops = args.articulation_loops
        for loop_idx in range(num_loops):
            if not articulation_success:
                break

            for wp_idx, (wp_pos, wp_quat) in enumerate(waypoints):
                data.mocap_pos[0] = wp_pos
                data.mocap_quat[0] = wp_quat

                mujoco.mj_step(model, data, nstep=10)
                if render and viewer is not None:
                    viewer.sync()

                if wp_idx % 10 == 0 and wp_idx < len(waypoints) * 0.9:
                    is_currently_grasping = is_grasping(model, data, handle_geoms)
                else:
                    is_currently_grasping = True

                joint_position = get_joint_position(model, data, primary_joint["name"])
                joint_positions.append(joint_position)

                if not is_currently_grasping:
                    articulation_success = False
                    break

            if not articulation_success:
                break

            for wp_idx in range(len(waypoints) - 1, -1, -1):
                wp_pos, wp_quat = waypoints[wp_idx]
                data.mocap_pos[0] = wp_pos
                data.mocap_quat[0] = wp_quat

                mujoco.mj_step(model, data, nstep=10)
                if render and viewer is not None:
                    viewer.sync()

                if wp_idx % 10 == 0 and wp_idx < len(waypoints) * 0.9:
                    is_currently_grasping = is_grasping(model, data, handle_geoms)
                else:
                    is_currently_grasping = True

                joint_position = get_joint_position(model, data, primary_joint["name"])
                joint_positions.append(joint_position)

                if not is_currently_grasping:
                    articulation_success = False
                    break

    if joint_positions:
        sufficient_movement = check_sufficient_joint_movement(joint_positions)
        if not sufficient_movement:
            articulation_success = False

    if render and viewer is not None:
        viewer.close()
        del viewer
        time.sleep(0.1)
    if articulation_success:
        return i, transform, quality
    else:
        return i, None, None


def run_simulation_with_viewer(
    model,
    data,
    xml_content,
    object_name,
    use_viewer,
    args,
    primary_joint=None,
    handle_geoms=None,
):
    with open(args.grasps_path, "r") as f:
        grasp_data = json.load(f)
    transforms = np.array(grasp_data["transforms"])
    print(len(transforms), "grasps to evaluate")
    qualities = np.array(grasp_data.get("quality_antipodal", [1.0] * len(transforms)))
    width = np.array(grasp_data.get("grasp_widths", [0.1] * len(transforms)))

    joint_info_override = primary_joint

    if use_viewer:
        successful_transforms = []
        successful_qualities = []
        successful_widths = []

        pbar = tqdm(
            enumerate(zip(transforms, qualities)),
            total=len(transforms),
            desc="Testing grasps (0/0 successful)",
        )
        for i, (transform, quality) in pbar:
            pos = transform[:3, 3]
            quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

            approach_distance = args.approach_distance
            approach_vector = transform[:3, 2] * approach_distance
            approach_pos = pos - approach_vector

            tree = ET.ElementTree(ET.fromstring(xml_content))
            root = tree.getroot()
            gripper_base = root.find(".//body[@name='base']")
            if gripper_base is not None:
                gripper_base.set(
                    "pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}"
                )
                gripper_base.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
            target_ee_pose = root.find(".//body[@name='target_ee_pose']")
            if target_ee_pose is not None:
                target_ee_pose.set(
                    "pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}"
                )
                target_ee_pose.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")

            for geom in root.findall(".//geom"):
                if geom.get("name", "").startswith("traj_sphere_"):
                    parent = geom.getparent() if hasattr(geom, "getparent") else None
                    if parent is not None:
                        parent.remove(geom)
                    else:
                        root.remove(geom)

            joint_info = joint_info_override
            if not joint_info or (
                isinstance(joint_info, dict)
                and "name" not in joint_info
                and "primary_joint" not in joint_info
            ):
                print(
                    f"[ERROR] No valid joint axis info for joint '{object_name}' or missing 'name'. Skipping this grasp."
                )
                continue

            i, transform_result, quality_result = test_single_grasp(
                (i, transform, quality, args),
                object_name,
                xml_content,
                args,
                primary_joint=joint_info_override,
                render=True,
                handle_geoms=handle_geoms,
            )

            if transform_result is not None:
                successful_transforms.append(transform_result)
                successful_qualities.append(quality_result)
                successful_widths.append(0.1)

            pbar.set_description(
                f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
            )

            # Check if we've reached max_successful grasps and should stop early
            if (
                args.max_successful > 0
                and len(successful_transforms) >= args.max_successful
            ):
                print(
                    f"\nReached maximum successful grasps ({args.max_successful}). Stopping early."
                )
                break

        return successful_transforms, successful_qualities, successful_widths

    else:
        config = {
            "approach_distance": args.approach_distance,
            "approach_steps": args.approach_steps,
        }

        grasp_params = [
            (i, transform, quality, config)
            for i, (transform, quality) in enumerate(zip(transforms, qualities))
        ]

        grasp_params_batch = grasp_params

        num_workers = min(args.num_workers, len(grasp_params_batch))

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
                            (i, transform_result, quality_result, 0.1)
                        )

                        if (
                            args.max_successful > 0
                            and success_count.value >= args.max_successful
                        ):
                            should_stop.value = True
                            tqdm.write(
                                f"Found {success_count.value} successful grasps (reached max_successful limit)"
                            )
                            # Return early to avoid more processing
                            return

                pbar.set_description(
                    f"Testing grasps ({success_count.value}/{processed_count.value} successful)"
                )
                pbar.update(1)

            pbar = tqdm(
                total=len(grasp_params_batch), desc="Testing grasps (0/0 successful)"
            )

            with mp.Pool(processes=num_workers) as pool:
                results = [
                    pool.apply_async(
                        test_single_grasp,
                        args=(
                            param,
                            object_name,
                            xml_content,
                            args,
                            handle_geoms,
                            primary_joint,
                        ),
                        callback=update_progress_bar,
                    )
                    for param in grasp_params_batch
                ]

                completed = 0
                while completed < len(results):
                    if should_stop.value:
                        pool.terminate()
                        tqdm.write(
                            "Terminating remaining workers after reaching max successful grasps"
                        )
                        # Force break immediately instead of waiting for all tasks
                        break

                    # Add a small delay to prevent high CPU usage during polling
                    time.sleep(0.01)

                    for i, r in enumerate(results):
                        if r is not None and r.ready() and not r.successful():
                            try:
                                r.get()
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
                else:
                    # When stopped early, close the pool without waiting
                    pool.close()

            successful_transforms_only = [t for _, t, _, _ in successful_transforms]
            successful_qualities_only = [q for _, _, q, _ in successful_transforms]
            successful_widths_only = [w for _, _, _, w in successful_transforms]

        pbar.close()
        return (
            successful_transforms_only,
            successful_qualities_only,
            successful_widths_only,
        )


def filter_per_joint_summary(summary_json_path, args):
    with open(summary_json_path, "r") as f:
        summary = json.load(f)
    updated_summary = []
    for entry in summary:
        joint = entry.get("joint")
        grasps_file = entry.get("grasps_file")
        handle_mesh = entry.get("handle_mesh")
        xml_file = entry.get("xml_file") if entry.get("xml_file") else args.xml_file
        joint_info = (
            entry.get("primary_joint")
            or entry.get("joint_axis")
            or entry.get("joint_info")
        )
        if not (joint and grasps_file and os.path.exists(grasps_file)):
            print(f"Skipping joint {joint}: missing grasps file {grasps_file}")
            continue
        filter_args = args
        filter_args.grasps_path = grasps_file
        filter_args.object_name = joint if joint else args.object_name
        filter_args.xml_file = xml_file
        try:
            main_single_file_filtering(
                grasps_file,
                filter_args.object_name,
                filter_args.xml_file,
                filter_args,
                joint_axis_info=joint_info,
            )
            entry["filtered_grasps_file"] = grasps_file.replace(
                ".json", "_filtered.json"
            )
        except Exception as e:
            print(f"  Error filtering grasps for joint {joint}: {e}")
            traceback.print_exc()
            entry["filtered_grasps_file"] = None
        updated_summary.append(entry)
    summary_out = summary_json_path.replace(".json", "_filtered.json")
    with open(summary_out, "w") as f:
        json.dump(updated_summary, f, indent=2)


def main_single_file_filtering(
    grasps_path, object_name, xml_file, args, joint_axis_info=None, handle_geoms=None
):
    xml_path = os.path.join(os.path.dirname(__file__), "../assets/scene.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    try:
        with open(xml_file, "r") as f:
            obj_xml_content = f.read()
        obj_tree = ET.fromstring(obj_xml_content)
        free_joints = obj_tree.findall(".//joint[@type='free']")
        if free_joints:
            for joint in free_joints:
                for parent in obj_tree.findall(".//*"):
                    for child in parent.findall("joint"):
                        if (
                            child.get("name") == joint.get("name")
                            and child.get("type") == "free"
                        ):
                            parent.remove(child)
                            print(f"Removed free joint: {joint.get('name')}")
        with open(xml_file, "w") as f:
            f.write(ET.tostring(obj_tree, encoding="unicode"))
        include = ET.Element("include", {"file": xml_file})
    except Exception as e:
        print(f"Error modifying XML to remove free joints: {e}")
        include = ET.Element("include", {"file": xml_file})
    root.append(include)
    worldbody = root.find("worldbody")
    if joint_axis_info is not None:
        primary_joint = joint_axis_info
    else:
        print(
            f"[ERROR] No joint axis info provided for joint '{object_name}'. Skipping all grasps for this joint."
        )
        return 0, None

    xml_content = ET.tostring(root, encoding="unicode")
    robot_xml_path = os.path.join(
        os.path.dirname(__file__),
        f"../assets/gripper_models/{args.gripper}_gripper/model_articulate.xml",
    )
    with open(robot_xml_path, "r") as f:
        robot_xml_content = f.read()
    xml_content = merge_xml_contents(xml_content, robot_xml_content)

    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    successful_transforms, successful_qualities, successful_widths = (
        run_simulation_with_viewer(
            model,
            data,
            xml_content,
            object_name,
            args.render,
            args,
            primary_joint=primary_joint,
            handle_geoms=handle_geoms,
        )
    )

    transforms_list = []
    for transform in successful_transforms:
        if isinstance(transform, np.ndarray):
            transform = transform.tolist()
        transforms_list.append(transform)
    qualities_list = []
    for quality in successful_qualities:
        if isinstance(quality, np.ndarray):
            quality = quality.tolist()
        qualities_list.append(quality)
    widths_list = []
    for width in successful_widths:
        if isinstance(width, np.ndarray):
            width = width.tolist()
        widths_list.append(width)

    output_path = grasps_path.replace(".json", "_filtered.json")
    with open(output_path, "w") as f:
        with open(grasps_path, "r") as original_f:
            original_data = json.load(original_f)
        json.dump(
            {
                "transforms": transforms_list,
                "quality_antipodal": qualities_list,
                "object": original_data.get("object", "unknown_object"),
                "object_scale": original_data.get("object_scale", 1.0),
                "object_position": original_data.get("object_position", [0, 0, 0]),
                "object_rotation": original_data.get("object_rotation", [1, 0, 0, 0]),
                "approach_distance": args.approach_distance,
                "grasp_widths": widths_list,
                "object_class": original_data.get("object_class", "unknown"),
                "object_dataset": original_data.get("object_dataset", "unknown"),
                "gripper": original_data.get("gripper", "unknown_gripper"),
                "gripper_configuration": original_data.get("gripper_configuration", []),
                "transforms_quality": original_data.get("transforms_quality", []),
                "roll_angles": original_data.get("roll_angles", []),
                "standoffs": original_data.get("standoffs", []),
                "mesh_points": original_data.get("mesh_points", []),
                "mesh_normals": original_data.get("mesh_normals", []),
                "collisions": original_data.get("collisions", []),
            },
            f,
            indent=2,
        )
    return len(successful_transforms), output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_name", type=str)
    parser.add_argument(
        "--grasps_path", type=str, help="Path to single grasps file (legacy mode)"
    )
    parser.add_argument("--gripper", type=str, default="robotiq")
    parser.add_argument("--xml_file", type=str)
    parser.add_argument(
        "--per_joint_summary_json",
        type=str,
        default=None,
        help="Path to summary JSON mapping joints to handle meshes and grasp files",
    )
    parser.add_argument("--approach_distance", type=float, default=0.1)
    parser.add_argument("--approach_steps", type=int, default=1000)
    parser.add_argument("--articulation_loops", type=int, default=1)
    parser.add_argument("--waypoint_pause", type=float, default=0.025)
    parser.add_argument("--endpoint_pause", type=float, default=0.2)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    parser.add_argument("--max_successful", type=int, default=0)
    parser.add_argument("--filtered", action="store_true")
    args = parser.parse_args()
    if args.per_joint_summary_json:
        summary_path = os.path.abspath(args.per_joint_summary_json)
        os.chdir(os.path.dirname(summary_path))
        with open(summary_path, "r") as f:
            summary = json.load(f)
        updated_summary = []
        for entry in summary:
            joint = entry.get("joint")
            if args.filtered:
                grasps_file = entry.get("filtered_grasps_file")
            else:
                grasps_file = entry.get("grasps_file")
            handle_mesh = entry.get("handle_mesh")
            xml_file = entry.get("xml_file") if entry.get("xml_file") else args.xml_file
            joint_info = (
                entry.get("primary_joint")
                or entry.get("joint_axis")
                or entry.get("joint_info")
            )
            handle_geoms = entry.get("handle_geoms", [])
            if not (joint and grasps_file and os.path.exists(grasps_file)):
                print(f"Skipping joint {joint}: missing grasps file {grasps_file}")
                continue
            filter_args = args
            filter_args.grasps_path = grasps_file
            filter_args.object_name = joint if joint else args.object_name
            filter_args.xml_file = xml_file

            try:
                main_single_file_filtering(
                    grasps_file,
                    filter_args.object_name,
                    filter_args.xml_file,
                    filter_args,
                    joint_axis_info=joint_info,
                    handle_geoms=handle_geoms,
                )
                entry["filtered_grasps_file"] = grasps_file.replace(
                    ".json", "_filtered.json"
                )
            except Exception as e:
                print(f"  Error filtering grasps for joint {joint}: {e}")
                traceback.print_exc()
                entry["filtered_grasps_file"] = None
            updated_summary.append(entry)
        summary_out = summary_path.replace(".json", "_filtered.json")
        with open(summary_out, "w") as f:
            json.dump(updated_summary, f, indent=2)
        return


if __name__ == "__main__":
    main()
