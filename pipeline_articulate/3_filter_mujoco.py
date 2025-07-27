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


parser = argparse.ArgumentParser()
parser.add_argument("--object_name", type=str)
parser.add_argument("--grasps_path", type=str)
parser.add_argument("--xml_file", type=str)
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
parser.add_argument(
    "--articulation_loops",
    type=int,
    default=1,
    help="Number of articulation loops (forward and backward) to perform",
)
parser.add_argument(
    "--waypoint_pause",
    type=float,
    default=0.025,
    help="Pause time in seconds between each waypoint",
)
parser.add_argument(
    "--endpoint_pause",
    type=float,
    default=0.2,
    help="Pause time in seconds at the endpoints of articulation",
)
parser.add_argument("--render", action="store_true", help="Enable interactive viewer")
parser.add_argument(
    "--num_workers",
    type=int,
    default=mp.cpu_count(),
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


def is_grasping(model, data, handle_name):
    left_patterns = ["left_finger", "finger_l", "gripper_finger_left"]
    right_patterns = ["right_finger", "finger_r", "gripper_finger_right"]

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        if not geom1 or not geom2:
            continue

        if handle_name.lower() in geom1.lower() or handle_name.lower() in geom2.lower():
            other = geom2 if handle_name.lower() in geom1.lower() else geom1
            if any(p in other.lower() for p in left_patterns) or any(
                p in other.lower() for p in right_patterns
            ):
                return True

        if (
            any(p in geom1.lower() for p in left_patterns)
            and any(p in geom2.lower() for p in right_patterns)
        ) or (
            any(p in geom2.lower() for p in left_patterns)
            and any(p in geom1.lower() for p in right_patterns)
        ):
            return False

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


def test_single_grasp(grasp_data, object_name, xml_content):
    joint_axis_path = os.path.join(
        os.path.dirname(args.grasps_path), f"{object_name}_joint_axis.json"
    )
    joint_info = None
    if os.path.exists(joint_axis_path):
        with open(joint_axis_path, "r") as f:
            joint_info = json.load(f)

    i, transform, quality, config = grasp_data
    pos = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

    approach_distance = config["approach_distance"]
    approach_steps = config["approach_steps"]
    approach_vector = transform[:3, 2] * approach_distance
    approach_pos = pos - approach_vector

    # Parse XML and update positions
    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()
    gripper_base = root.find(".//body[@name='gripper_base']")
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

    # Initialize simulation
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)

    # Open gripper
    data.ctrl[0] = 1.0

    for _ in range(500):
        mujoco.mj_step(model, data)

    # Approach object
    joint_position_before = get_joint_position(
        model, data, joint_info["primary_joint"]["name"]
    )

    for step in range(approach_steps):
        new_pos = approach_pos + (step / approach_steps) * approach_vector
        data.mocap_pos[0] = new_pos
        data.mocap_quat[0] = quat
        mujoco.mj_step(model, data, nstep=1000)

    joint_position_after = get_joint_position(
        model, data, joint_info["primary_joint"]["name"]
    )

    if joint_position_before is None or joint_position_after is None:
        return i, None, None
    if np.abs(joint_position_after - joint_position_before) > 0.01:
        return i, None, None

    # Move to final position
    data.mocap_pos[0] = pos
    mujoco.mj_step(model, data, nstep=1000)

    # Close gripper
    data.ctrl[0] = -1.0
    mujoco.mj_step(model, data, nstep=1000)

    # Check if object is grasped
    if not is_grasping(model, data, object_name):
        return i, None, None

    # Load joint information

    # Setup waypoints for articulation
    num_waypoints = 200
    waypoints = []
    if joint_info and "primary_joint" in joint_info:
        primary_joint = joint_info["primary_joint"]
        joint_type = primary_joint.get("type")
        joint_range_str = primary_joint.get("range", "0 0")
        joint_range = [float(x) for x in joint_range_str.split()]

        gripper_pos = data.body("gripper_base").xpos.copy()
        gripper_quat = data.body("gripper_base").xquat.copy()

        if joint_type == "hinge":
            rotation_axis = primary_joint.get("rotation_axis", {"x": 0, "y": 0, "z": 0})
            axis_world = np.array(
                [
                    rotation_axis.get("x", 0),
                    rotation_axis.get("y", 0),
                    rotation_axis.get("z", 0),
                ]
            )
            joint_position = primary_joint.get("position", {"x": 0, "y": 0, "z": 0})
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
            axis_str = primary_joint.get("axis", "0 0 0")
            slide_axis = np.array([float(x) for x in axis_str.split()])
            max_distance = joint_range[1]
            if max_distance == 0:
                max_distance = 0.2
            for i in range(num_waypoints + 1):
                distance = i * max_distance / num_waypoints
                new_pos = gripper_pos + slide_axis * distance
                waypoints.append((new_pos, gripper_quat.copy()))

    # Execute articulation
    joint_positions = []
    articulation_success = True

    if waypoints:
        num_loops = args.articulation_loops
        for loop_idx in range(num_loops):
            if not articulation_success:
                break

            # Forward movement
            for wp_idx, (wp_pos, wp_quat) in enumerate(waypoints):
                data.mocap_pos[0] = wp_pos
                data.mocap_quat[0] = wp_quat

                mujoco.mj_step(model, data, nstep=200)

                is_currently_grasping = is_grasping(model, data, object_name)
                joint_position = get_joint_position(model, data, primary_joint["name"])
                joint_positions.append(joint_position)

                if not is_currently_grasping:
                    articulation_success = False
                    break

            if not articulation_success:
                break

            # Backward movement
            for wp_idx in range(len(waypoints) - 1, -1, -1):
                wp_pos, wp_quat = waypoints[wp_idx]
                data.mocap_pos[0] = wp_pos
                data.mocap_quat[0] = wp_quat

                mujoco.mj_step(model, data, nstep=200)

                is_currently_grasping = is_grasping(model, data, object_name)
                joint_position = get_joint_position(model, data, primary_joint["name"])
                joint_positions.append(joint_position)

                if not is_currently_grasping:
                    articulation_success = False
                    break

    # Check if there was sufficient joint movement
    if joint_positions:
        sufficient_movement = check_sufficient_joint_movement(joint_positions)
        if not sufficient_movement:
            articulation_success = False

    # Return results
    if articulation_success:
        return i, transform, quality
    else:
        return i, None, None


def run_simulation_with_viewer(model, data, xml_content, object_name, use_viewer):
    with open(args.grasps_path, "r") as f:
        grasp_data = json.load(f)
    transforms = np.array(grasp_data["transforms"])
    qualities = np.array(grasp_data.get("quality_antipodal", [1.0] * len(transforms)))

    width = np.array(grasp_data.get("grasp_widths", [0.1] * len(transforms)))

    if use_viewer:
        successful_transforms = []
        successful_qualities = []
        successful_widths = []

        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=True, show_right_ui=True
        ) as viewer:
            pbar = tqdm(
                enumerate(zip(transforms, qualities)),
                total=len(transforms),
                desc="Testing grasps (0/0 successful)",
            )

            for i, (transform, quality) in pbar:
                if i < 50:
                    continue
                pos = transform[:3, 3]
                quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

                approach_distance = args.approach_distance
                approach_vector = transform[:3, 2] * approach_distance
                approach_pos = pos - approach_vector

                tree = ET.ElementTree(ET.fromstring(xml_content))
                root = tree.getroot()
                gripper_base = root.find(".//body[@name='gripper_base']")
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
                    target_ee_pose.set(
                        "quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
                    )

                for geom in root.findall(".//geom"):
                    if geom.get("name", "").startswith("traj_sphere_"):
                        parent = (
                            geom.getparent() if hasattr(geom, "getparent") else None
                        )
                        if parent is not None:
                            parent.remove(geom)
                        else:
                            root.remove(geom)

                joint_axis_path = os.path.join(
                    os.path.dirname(args.grasps_path),
                    f"{args.object_name}_joint_axis.json",
                )
                joint_info = None
                if os.path.exists(joint_axis_path):
                    with open(joint_axis_path, "r") as f:
                        joint_info = json.load(f)
                waypoints = []
                num_waypoints = 200
                if joint_info and "primary_joint" in joint_info:
                    primary_joint = joint_info["primary_joint"]
                    joint_type = primary_joint.get("type")
                    joint_range_str = primary_joint.get("range", "0 0")
                    joint_range = [float(x) for x in joint_range_str.split()]
                    gripper_pos = approach_pos + approach_vector
                    gripper_quat = quat
                    if joint_type == "hinge":
                        rotation_axis = primary_joint.get(
                            "rotation_axis", {"x": 0, "y": 0, "z": 0}
                        )
                        axis_world = np.array(
                            [
                                rotation_axis.get("x", 0),
                                rotation_axis.get("y", 0),
                                rotation_axis.get("z", 0),
                            ]
                        )
                        joint_position = primary_joint.get(
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
                            rotation_matrix = rotation_matrix_from_axis_angle(
                                axis_world, angle
                            )
                            new_rel_pos = rotate_vector(rel_pos, rotation_matrix)
                            new_pos = pivot_point + new_rel_pos
                            waypoints.append((new_pos, gripper_quat))
                    elif joint_type == "slide":
                        axis_str = primary_joint.get("axis", "0 0 0")
                        slide_axis = np.array([float(x) for x in axis_str.split()])
                        max_distance = joint_range[1]
                        if max_distance == 0:
                            max_distance = 0.2
                        for i in range(num_waypoints + 1):
                            distance = i * max_distance / num_waypoints
                            new_pos = gripper_pos + slide_axis * distance
                            waypoints.append((new_pos, gripper_quat))

                worldbody = root.find("worldbody")
                if worldbody is not None and waypoints:
                    for idx, (wp_pos, _) in enumerate(waypoints):
                        sphere = ET.Element(
                            "geom",
                            {
                                "name": f"traj_sphere_{idx}",
                                "type": "sphere",
                                "size": "0.01",
                                "rgba": "0 0 1 0.5",
                                "pos": f"{wp_pos[0]} {wp_pos[1]} {wp_pos[2]}",
                                "contype": "0",
                                "conaffinity": "0",
                            },
                        )
                        worldbody.append(sphere)

                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                data = mujoco.MjData(model)

                viewer.close()
                time.sleep(0.1)
                del viewer
                viewer = mujoco.viewer.launch_passive(
                    model, data, show_left_ui=False, show_right_ui=False
                )

                data.ctrl[0] = 1.0

                for step in range(500):
                    mujoco.mj_step(model, data)
                    if step % 50 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            return (
                                successful_transforms,
                                successful_qualities,
                                successful_widths,
                            )

                approach_steps = args.approach_steps

                current_time = time.time()
                joint_position_before = get_joint_position(
                    model, data, joint_info["primary_joint"]["name"]
                )
                for step in range(approach_steps):
                    while time.time() - current_time < 1.0:
                        pass
                    mujoco.mj_step(model, data, nstep=1000)
                    viewer.sync()
                    current_time = time.time()
                    new_pos = approach_pos + (step / approach_steps) * approach_vector
                    data.mocap_pos[0] = new_pos
                    data.mocap_quat[0] = quat

                joint_position_after = get_joint_position(
                    model, data, joint_info["primary_joint"]["name"]
                )

                if joint_position_before is None or joint_position_after is None:
                    continue

                if np.abs(joint_position_after - joint_position_before) > 0.01:
                    print(
                        "Unwanted joint movement during approach, skipping this grasp"
                    )
                    continue

                data.mocap_pos[0] = pos
                for _ in range(1000):
                    mujoco.mj_step(model, data)
                    viewer.sync()

                for v in range(1000):
                    data.ctrl[0] = (500.0-v) / 500.0
                    mujoco.mj_step(model, data)
                    viewer.sync()

                for _ in range(1000):
                    mujoco.mj_step(model, data)
                    viewer.sync()

                joint_axis_path = os.path.join(
                    os.path.dirname(args.grasps_path),
                    f"{args.object_name}_joint_axis.json",
                )
                joint_info = None

                if os.path.exists(joint_axis_path):
                    with open(joint_axis_path, "r") as f:
                        joint_info = json.load(f)
                else:
                    print(
                        f"Warning: No joint axis information found at {joint_axis_path}"
                    )

                # Check if object is grasped
                if not is_grasping(model, data, object_name):
                    print(
                        "Grasp is not successful before articulation, skipping this grasp"
                    )
                    pbar.set_description(
                        f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                    )
                    continue

                num_waypoints = 200
                waypoints = []

                if joint_info and "primary_joint" in joint_info:
                    primary_joint = joint_info["primary_joint"]
                    joint_type = primary_joint.get("type")
                    joint_range_str = primary_joint.get("range", "0 0")
                    joint_range = [float(x) for x in joint_range_str.split()]

                    # We don't need object_body or object_pos, but keeping for consistency with test_single_grasp
                    # object_body = data.body(args.object_name)

                    gripper_pos = data.body("gripper_base").xpos.copy()
                    gripper_quat = data.body("gripper_base").xquat.copy()

                    if joint_type == "hinge":
                        rotation_axis = primary_joint.get(
                            "rotation_axis", {"x": 0, "y": 0, "z": 0}
                        )
                        axis_world = np.array(
                            [
                                rotation_axis.get("x", 0),
                                rotation_axis.get("y", 0),
                                rotation_axis.get("z", 0),
                            ]
                        )

                        joint_position = primary_joint.get(
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

                            rotation_matrix = rotation_matrix_from_axis_angle(
                                axis_world, angle
                            )

                            new_rel_pos = rotate_vector(rel_pos, rotation_matrix)
                            new_pos = pivot_point + new_rel_pos

                            rotation_quat = axis_angle_to_quat(axis_world, angle)
                            new_quat = quat_multiply(rotation_quat, gripper_quat)

                            waypoints.append((new_pos, new_quat))

                    elif joint_type == "slide":
                        axis_str = primary_joint.get("axis", "0 0 0")
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

                if waypoints:
                    num_loops = args.articulation_loops

                    for loop_idx in range(num_loops):
                        if not articulation_success:
                            break

                        for wp_idx, (wp_pos, wp_quat) in enumerate(waypoints):
                            data.mocap_pos[0] = wp_pos
                            data.mocap_quat[0] = wp_quat

                            for step in range(200):
                                mujoco.mj_step(model, data)
                                if step % 10 == 0:
                                    viewer.sync()
                                    if not viewer.is_running():
                                        return (
                                            successful_transforms,
                                            successful_qualities,
                                            successful_widths,
                                        )

                            is_currently_grasping = is_grasping(
                                model, data, object_name
                            )
                            joint_position = get_joint_position(
                                model, data, primary_joint["name"]
                            )
                            joint_positions.append(joint_position)

                            if not is_currently_grasping:
                                print(
                                    "Lost grasp during articulation, marking as failed"
                                )
                                articulation_success = False
                                break

                        if not articulation_success:
                            break

                        for wp_idx in range(len(waypoints) - 1, -1, -1):
                            wp_pos, wp_quat = waypoints[wp_idx]

                            data.mocap_pos[0] = wp_pos
                            data.mocap_quat[0] = wp_quat

                            for step in range(200):
                                mujoco.mj_step(model, data)
                                if step % 10 == 0:
                                    viewer.sync()
                                    if not viewer.is_running():
                                        return (
                                            successful_transforms,
                                            successful_qualities,
                                            successful_widths,
                                        )

                            is_currently_grasping = is_grasping(
                                model, data, object_name
                            )
                            joint_position = get_joint_position(
                                model, data, primary_joint["name"]
                            )
                            joint_positions.append(joint_position)

                            if not is_currently_grasping:
                                print(
                                    "Lost grasp during articulation, marking as failed"
                                )
                                articulation_success = False
                                break

                    sufficient_movement = check_sufficient_joint_movement(
                        joint_positions
                    )
                    if not sufficient_movement:
                        print(
                            "Insufficient joint movement during articulation, marking as failed"
                        )
                        articulation_success = False

                    if articulation_success:
                        print("Completed all articulation loops successfully")
                        # If articulation was successful, add this grasp to our successes
                        successful_transforms.append(transform)
                        successful_qualities.append(quality)
                        successful_widths.append(0.1)
                        pbar.set_description(
                            f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                        )
                    else:
                        print("Articulation failed, skipping this grasp")
                        pbar.set_description(
                            f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                        )
                        continue

                # Skip processing if articulation was necessary but failed
                if (
                    joint_info
                    and "primary_joint" in joint_info
                    and waypoints
                    and not articulation_success
                ):
                    pbar.set_description(
                        f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                    )
                    continue

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

                pbar.set_description(
                    f"Testing grasps ({success_count.value}/{processed_count.value} successful)"
                )
                pbar.update(1)

            pbar = tqdm(total=len(grasp_params), desc="Testing grasps (0/0 successful)")

            with mp.Pool(processes=num_workers) as pool:
                results = [
                    pool.apply_async(
                        test_single_grasp,
                        args=(param, object_name, xml_content),
                        callback=update_progress_bar,
                    )
                    for param in grasp_params
                ]

                completed = 0
                while completed < len(results):
                    if should_stop.value:
                        pool.terminate()
                        tqdm.write(
                            "Terminating remaining workers after reaching max successful grasps"
                        )
                        break

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

            successful_transforms_only = [t for _, t, _, _ in successful_transforms]
            successful_qualities_only = [q for _, _, q, _ in successful_transforms]
            successful_widths_only = [w for _, _, _, w in successful_transforms]

        pbar.close()
        return (
            successful_transforms_only,
            successful_qualities_only,
            successful_widths_only,
        )


if __name__ == "__main__":
    xml_path = os.path.join(os.path.dirname(__file__), "../assets/scene.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_name = args.object_name

    try:
        with open(args.xml_file, "r") as f:
            obj_xml_content = f.read()

        obj_tree = ET.fromstring(obj_xml_content)

        free_joints = obj_tree.findall(".//joint[@type='free']")
        if free_joints:
            print(f"Found {len(free_joints)} free joints to remove")
            for joint in free_joints:
                for parent in obj_tree.findall(".//*"):
                    for child in parent.findall("joint"):
                        if (
                            child.get("name") == joint.get("name")
                            and child.get("type") == "free"
                        ):
                            parent.remove(child)
                            print(f"Removed free joint: {joint.get('name')}")

        with open(args.xml_file, "w") as f:
            f.write(ET.tostring(obj_tree, encoding="unicode"))

        include = ET.Element("include", {"file": args.xml_file})
    except Exception as e:
        print(f"Error modifying XML to remove free joints: {e}")

        include = ET.Element("include", {"file": args.xml_file})

    root.append(include)

    worldbody = root.find("worldbody")

    joint_axis_file = args.grasps_path.replace("_grasps.json", "_joint_axis.json")
    joint_axis_file = args.grasps_path.replace(
        "_grasps_filtered.json", "_joint_axis.json"
    )

    print(f"Using joint axis file: {joint_axis_file}")
    if os.path.exists(joint_axis_file):
        try:
            with open(joint_axis_file, "r") as f:
                joint_data = json.load(f)

            primary_joint = joint_data.get("primary_joint")
            if primary_joint:
                if (
                    "parent_position" in primary_joint
                    and "parent_rotation" in primary_joint
                ):
                    parent_pos = primary_joint["parent_position"]
                    parent_rot = primary_joint["parent_rotation"]

                    global_pos = [parent_pos["x"], parent_pos["y"], parent_pos["z"]]

                    cylinder_pos = f"{global_pos[0]} {global_pos[1]} {global_pos[2]}"

                    rotation_axis = primary_joint.get(
                        "rotation_axis", {"x": 0, "y": 1, "z": 0}
                    )
                    global_axis = [
                        rotation_axis["x"],
                        rotation_axis["y"],
                        rotation_axis["z"],
                    ]

                    default_axis = np.array([0, 0, 1])
                    joint_axis_vec = np.array(global_axis)

                    joint_axis_norm = np.linalg.norm(joint_axis_vec)
                    if joint_axis_norm > 0:
                        joint_axis_vec = joint_axis_vec / joint_axis_norm

                    if np.allclose(default_axis, joint_axis_vec):
                        quat = [1, 0, 0, 0]
                    elif np.allclose(default_axis, -joint_axis_vec):
                        quat = [0, 1, 0, 0]
                    else:
                        from scipy.spatial.transform import Rotation as R_scipy

                        cross_product = np.cross(default_axis, joint_axis_vec)
                        dot_product = np.dot(default_axis, joint_axis_vec)

                        if np.linalg.norm(cross_product) > 1e-6:
                            rotation = R_scipy.align_vectors(
                                [joint_axis_vec], [default_axis]
                            )[0]
                            quat = rotation.as_quat(scalar_first=True)
                        else:
                            quat = [1, 0, 0, 0]

                    quat_str = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"

                    joint_axis_info = {
                        "position": np.array(global_pos),
                        "axis": np.array(global_axis),
                        "axis_normalized": joint_axis_vec,
                    }

                else:
                    joint_axis_info = None

            else:
                joint_axis_info = None
                print("Warning: No primary joint found in joint analysis")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            joint_axis_info = None
            print(f"Warning: Could not load joint axis file {joint_axis_file}: {e}")
        except Exception as e:
            joint_axis_info = None
            print(f"Warning: Error processing joint axis data: {e}")
    else:
        joint_axis_info = None
        print(f"Warning: Joint axis file not found: {joint_axis_file}")
        print(
            "Run Stage 1 (joint axis analysis) first to generate joint axis visualization"
        )

    xml_content = ET.tostring(root, encoding="unicode")

    robot_xml_path = os.path.join(
        os.path.dirname(__file__),
        "../assets/gripper_models/rum_gripper/model_articulate.xml",
    )
    with open(robot_xml_path, "r") as f:
        robot_xml_content = f.read()
    xml_content = merge_xml_contents(xml_content, robot_xml_content)

    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    successful_transforms, successful_qualities, successful_widths = (
        run_simulation_with_viewer(model, data, xml_content, object_name, args.render)
    )

    # Convert NumPy arrays to Python lists for JSON serialization
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

    output_path = args.grasps_path.replace(".json", "_filtered.json")
    with open(output_path, "w") as f:
        with open(args.grasps_path, "r") as original_f:
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

    tqdm.write(f"Saved {len(successful_transforms)} successful grasps to {output_path}")

    if not args.render and args.num_workers > 1:
        tqdm.write(f"Used {args.num_workers} parallel workers for grasp testing")
