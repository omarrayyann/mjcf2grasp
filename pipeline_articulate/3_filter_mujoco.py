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

parser = argparse.ArgumentParser()
parser.add_argument("--object_name", type=str)
parser.add_argument("--grasps_path", type=str)
parser.add_argument("--xml_file", type=str)
parser.add_argument("--num_shakes", type=int, default=2)
parser.add_argument("--shake_magnitude", type=float, default=0.1)
parser.add_argument("--shake_steps", type=int, default=500)
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
    default=3,
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
    default=0.5,
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
    return data.joint(joint_id).qpos


def check_sufficient_joint_movement(
    joint_positions, min_movement=0.0001, window_size=4
):
    """
    Check if there's sufficient joint movement across a sliding window of positions.

    Args:
        joint_positions: List of joint position values recorded at waypoints
        min_movement: Minimum required change to consider movement sufficient
        window_size: Number of waypoints to check across

    Returns:
        True if there's sufficient movement, False otherwise
    """
    if len(joint_positions) < window_size:
        return True  # Not enough data points to make determination

    for i in range(len(joint_positions) - window_size + 1):
        window = joint_positions[i : i + window_size]
        movement = max(window) - min(window)
        if movement < min_movement:
            print(
                f"Insufficient joint movement detected: {movement} (below {min_movement})"
            )
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
    """
    Merges two XML contents by copying elements from the additional XML into the base XML
    under the appropriate mujoco tags.

    Args:
        base_xml_content (str): The base XML content as a string
        additional_xml_content (str): The additional XML content to merge into the base

    Returns:
        str: The merged XML content as a string
    """
    # Parse both XML contents
    base_root = ET.fromstring(base_xml_content)
    additional_root = ET.fromstring(additional_xml_content)

    # Make sure both are mujoco elements
    if base_root.tag != "mujoco" or additional_root.tag != "mujoco":
        raise ValueError("Both XML contents must have 'mujoco' as the root element")

    # Dictionary to keep track of sections we've already processed
    processed_sections = {}

    # Process each child of additional_root
    for additional_child in additional_root:
        tag_name = additional_child.tag

        # Find the corresponding section in the base XML
        base_section = base_root.find(tag_name)

        # If section exists in base, merge the contents
        if base_section is not None:
            # Skip if we've already processed this section
            if tag_name in processed_sections:
                continue

            # Copy all elements from additional to base
            for element in additional_child:
                # Check if element with same name/attributes already exists to avoid duplicates
                is_duplicate = False
                for existing in base_section:
                    if element.tag == existing.tag and all(
                        attr in existing.attrib and existing.attrib[attr] == val
                        for attr, val in element.attrib.items()
                        if attr != "name"
                    ):
                        # For elements with name attribute, check if names match
                        if "name" in element.attrib and "name" in existing.attrib:
                            if element.attrib["name"] == existing.attrib["name"]:
                                is_duplicate = True
                                break
                        else:
                            is_duplicate = True
                            break

                # Add if not a duplicate
                if not is_duplicate:
                    base_section.append(element)

            processed_sections[tag_name] = True
        else:
            # If section doesn't exist in base, add it
            base_root.append(additional_child)
            processed_sections[tag_name] = True

    # Convert the merged XML back to string
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


def test_single_grasp(grasp_data, object_name):
    """Test a single grasp in a separate process"""
    i, transform, quality, config = grasp_data

    xml_path = os.path.join(os.path.dirname(__file__), "../assets/scene.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    include = ET.Element("include", {"file": args.xml_file})
    root.append(include)

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

    global initial_relative_position, initial_grasp_verified
    initial_relative_position = None
    initial_grasp_verified = False

    mujoco.mj_resetData(model, data)

    pos = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

    approach_distance = config.get("approach_distance", 0.1)
    approach_vector = transform[:3, 2] * approach_distance
    approach_pos = pos - approach_vector

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    model.body_pos[body_id] = approach_pos
    model.body_quat[body_id] = quat

    data.ctrl[0] = 1.0

    for step in range(500):
        mujoco.mj_step(model, data)

    approach_steps = config.get("approach_steps", 1000)
    velocity = approach_distance / (approach_steps * model.opt.timestep)
    for step in range(approach_steps):
        data.ctrl[2] = velocity
        mujoco.mj_step(model, data)

        current_pos = data.body("gripper_base").xpos
        distance_to_target = np.linalg.norm(current_pos - pos)
        if distance_to_target < 0.001:
            break

    data.ctrl[1] = 0.0
    data.ctrl[2] = 0.0
    data.ctrl[3] = 0.0

    for step in range(100):
        mujoco.mj_step(model, data)

    data.ctrl[0] = -1.0

    for step in range(1000):
        mujoco.mj_step(model, data)

    for step in range(2000):
        mujoco.mj_step(model, data)

    if not check_grasp(model, data, object_name, store_initial=True):
        return i, None, None

    directions = ["x", "y", "z"]
    shake_success = True

    for direction_idx, direction in enumerate(directions):
        ctrl_idx = direction_idx + 1

        for shake in range(config["num_shakes"]):
            total_steps = config["shake_steps"] * 2

            for step in range(total_steps):
                angle = 2 * np.pi * step / total_steps
                position = config["shake_magnitude"] * np.sin(angle)
                data.ctrl[ctrl_idx] = position

                mujoco.mj_step(model, data)

                if step == total_steps // 4 or step == 3 * total_steps // 4:
                    grasp_maintained = check_grasp(model, data, object_name)
                    if not grasp_maintained:
                        shake_success = False
                        break

            if not shake_success:
                break

            data.ctrl[ctrl_idx] = 0
            for step in range(50):
                mujoco.mj_step(model, data)

        data.ctrl[ctrl_idx] = 0

        if not shake_success:
            break

    final_grasp_check = is_object_grasped(model, data, object_name)

    if shake_success and final_grasp_check:
        return i, transform.tolist(), quality
    else:
        return i, None, None


def run_simulation_with_viewer(model, data, xml_content, object_name, use_viewer):
    """Run the simulation with or without interactive viewer"""

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
                desc=f"Testing grasps (0/0 successful)",
            )

            for i, (transform, quality) in pbar:
                if i < 646:  # 646
                    continue
                pos = transform[:3, 3]
                quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

                approach_distance = args.approach_distance
                approach_vector = transform[:3, 2] * approach_distance
                approach_pos = pos - approach_vector

                # change position od body gripper_base in xml

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

                geom_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, "test_sphere"
                )
                model.geom_pos[geom_id] = pos
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
                velocity = approach_distance / (approach_steps * model.opt.timestep)

                current_time = time.time()
                for step in range(approach_steps):
                    while time.time() - current_time < 1.0:
                        mujoco.mj_step(model, data)
                        viewer.sync()
                    current_time = time.time()
                    print("update")
                    new_pos = approach_pos + (step / approach_steps) * approach_vector
                    data.mocap_pos[0] = new_pos
                    data.mocap_quat[0] = quat

                data.mocap_pos[0] = pos
                for _ in range(1000):
                    mujoco.mj_step(model, data)
                    viewer.sync()

                data.ctrl[0] = -1.0

                for _ in range(1000):
                    mujoco.mj_step(model, data)
                    viewer.sync()
                print("Gripper closed")
                time.sleep(2.0)

                # HERE
                # Helper functions for quaternion operations since we don't want to rely on trimesh.transformations
                def rotation_matrix_from_axis_angle(axis, angle):
                    """Create a rotation matrix from axis and angle."""
                    axis = axis / np.linalg.norm(axis)  # Normalize axis
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
                    """Rotate a vector using a rotation matrix."""
                    return np.dot(rotation_matrix, vector)

                def quat_multiply(q1, q2):
                    """Multiply two quaternions."""
                    w1, x1, y1, z1 = q1
                    w2, x2, y2, z2 = q2
                    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
                    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
                    return np.array([w, x, y, z])

                def axis_angle_to_quat(axis, angle):
                    """Convert axis-angle to quaternion."""
                    axis = axis / np.linalg.norm(axis)  # Normalize axis
                    half_angle = angle * 0.5
                    s = np.sin(half_angle)
                    return np.array(
                        [np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s]
                    )

                # Load joint axis information for articulation
                joint_axis_path = os.path.join(
                    os.path.dirname(args.grasps_path),
                    f"{args.object_name}_joint_axis.json",
                )

                joint_info = None

                if os.path.exists(joint_axis_path):
                    with open(joint_axis_path, "r") as f:
                        joint_info = json.load(f)
                    print(f"Loaded joint axis information from: {joint_axis_path}")
                else:
                    print(
                        f"Warning: No joint axis information found at {joint_axis_path}"
                    )

                # Calculate articulation waypoints
                num_waypoints = 200
                waypoints = []

                if joint_info and "primary_joint" in joint_info:
                    primary_joint = joint_info["primary_joint"]
                    joint_type = primary_joint.get("type")
                    joint_range_str = primary_joint.get("range", "0 0")
                    joint_range = [float(x) for x in joint_range_str.split()]

                    # Get the position and orientation of the object
                    object_body = data.body(args.object_name)
                    object_pos = object_body.xpos.copy()

                    # Current gripper position and orientation
                    gripper_pos = data.body("gripper_base").xpos.copy()
                    gripper_quat = data.body("gripper_base").xquat.copy()

                    print(
                        f"Object position: {object_pos}, gripper position: {gripper_pos}"
                    )
                    print(f"Joint type: {joint_type}")

                    # Calculate waypoints based on joint type
                    if joint_type == "hinge":
                        # For a hinge joint, we'll rotate around the axis
                        # Get the rotation axis in world coordinates
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

                        # Get joint position in world coordinates
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

                        # Use range to determine how far to rotate
                        max_angle = joint_range[
                            1
                        ]  # Use the upper limit of the joint range
                        if max_angle == 0:
                            max_angle = (
                                np.pi / 2
                            )  # Default to 90 degrees if range is not specified

                        # Get the relative position of gripper from pivot point
                        rel_pos = gripper_pos - pivot_point

                        # Calculate waypoints for rotating around the axis
                        for i in range(num_waypoints + 1):
                            angle = i * max_angle / num_waypoints

                            # Create rotation matrix for this angle
                            rotation_matrix = rotation_matrix_from_axis_angle(
                                axis_world, angle
                            )

                            # Rotate the relative position and add back the pivot point
                            new_rel_pos = rotate_vector(rel_pos, rotation_matrix)
                            new_pos = pivot_point + new_rel_pos

                            # Calculate new orientation - gripper should follow the rotation
                            rotation_quat = axis_angle_to_quat(axis_world, angle)
                            new_quat = quat_multiply(rotation_quat, gripper_quat)

                            waypoints.append((new_pos, new_quat))
                            print(f"Waypoint {i}: pos={new_pos}, angle={angle}")

                    elif joint_type == "slide":
                        # For a slide joint, we'll move along the axis
                        # Get the slide direction in world coordinates
                        axis_str = primary_joint.get("axis", "0 0 0")
                        slide_axis = np.array([float(x) for x in axis_str.split()])

                        max_distance = joint_range[
                            1
                        ]  # Use the upper limit of the joint range
                        if max_distance == 0:
                            max_distance = (
                                0.2  # Default to 20cm if range is not specified
                            )

                        # Calculate waypoints for sliding along the axis
                        for i in range(num_waypoints + 1):
                            distance = i * max_distance / num_waypoints
                            new_pos = gripper_pos + slide_axis * distance
                            waypoints.append(
                                (new_pos, gripper_quat.copy())
                            )  # Maintain the same orientation
                            print(f"Waypoint {i}: pos={new_pos}, distance={distance}")

                # Execute articulation movement through waypoints
                time.sleep(1.0)

                # Check if grasp is successful before attempting articulation
                if not is_grasping(model, data, object_name):
                    print(
                        "Grasp is not successful before articulation, skipping this grasp"
                    )
                    pbar.set_description(
                        f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                    )
                    continue

                # Track joint positions during articulation to check for sufficient movement
                joint_positions = []
                articulation_success = True

                if waypoints:
                    # Use command-line argument for the number of loops
                    num_loops = args.articulation_loops

                    print(
                        f"Moving through {len(waypoints)} articulation waypoints for {num_loops} loops"
                    )

                    for loop_idx in range(num_loops):
                        if not articulation_success:
                            break

                        print(f"Loop {loop_idx + 1}/{num_loops}")

                        # Forward direction (0 to end)
                        print("Forward articulation movement...")
                        for wp_idx, (wp_pos, wp_quat) in enumerate(waypoints):
                            print(
                                f"Moving to waypoint {wp_idx + 1}/{len(waypoints)} (forward)"
                            )

                            # Set the target position
                            data.mocap_pos[0] = wp_pos
                            data.mocap_quat[0] = wp_quat

                            # Run simulation for a fixed number of steps for each waypoint
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
                            time.sleep(args.waypoint_pause)

                            # Check if still grasping
                            is_currently_grasping = is_grasping(
                                model, data, object_name
                            )
                            joint_position = get_joint_position(
                                model, data, primary_joint["name"]
                            )
                            joint_positions.append(joint_position)

                            print(
                                f"Grasping at waypoint {wp_idx + 1}: {is_currently_grasping}"
                            )
                            print(
                                f"Joint position for {primary_joint['name']}: {joint_position}"
                            )

                            # If grip is lost, mark as failed and exit loop
                            if not is_currently_grasping:
                                print(
                                    "Lost grasp during articulation, marking as failed"
                                )
                                articulation_success = False
                                break

                        if not articulation_success:
                            break

                        # Brief pause at the end position
                        time.sleep(args.endpoint_pause)

                        # Backward direction (end to 0)
                        print("Backward articulation movement...")
                        for wp_idx in range(len(waypoints) - 1, -1, -1):
                            wp_pos, wp_quat = waypoints[wp_idx]
                            print(
                                f"Moving to waypoint {len(waypoints) - wp_idx}/{len(waypoints)} (backward)"
                            )

                            # Set the target position
                            data.mocap_pos[0] = wp_pos
                            data.mocap_quat[0] = wp_quat

                            # Run simulation for a fixed number of steps for each waypoint
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
                            time.sleep(args.waypoint_pause)

                            # Check if still grasping
                            is_currently_grasping = is_grasping(
                                model, data, object_name
                            )
                            joint_position = get_joint_position(
                                model, data, primary_joint["name"]
                            )
                            joint_positions.append(joint_position)

                            print(
                                f"Grasping at waypoint {wp_idx + 1}: {is_currently_grasping}"
                            )
                            print(
                                f"Joint position for {primary_joint['name']}: {joint_position}"
                            )

                            # If grip is lost, mark as failed and exit loop
                            if not is_currently_grasping:
                                print(
                                    "Lost grasp during articulation, marking as failed"
                                )
                                articulation_success = False
                                break

                        # Brief pause at the initial position before next loop
                        time.sleep(args.endpoint_pause)

                    # Check if there was sufficient joint movement across the articulation
                    sufficient_movement = check_sufficient_joint_movement(
                        joint_positions
                    )
                    if not sufficient_movement:
                        print(
                            "Insufficient joint movement during articulation, marking as failed"
                        )
                        articulation_success = False

                    # Print min and max joint positions to help with debugging
                    if joint_positions:
                        min_pos = min(joint_positions)
                        max_pos = max(joint_positions)
                        total_movement = max_pos - min_pos
                        print(
                            f"Total joint movement: {total_movement} (min: {min_pos}, max: {max_pos})"
                        )

                    if articulation_success:
                        print("Completed all articulation loops successfully")
                    else:
                        print("Articulation failed, skipping this grasp")
                        pbar.set_description(
                            f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                        )
                        continue
                else:
                    print("No articulation waypoints calculated")

                # If we have joint info but articulation failed, skip this grasp
                if (
                    joint_info
                    and "primary_joint" in joint_info
                    and waypoints
                    and not articulation_success
                ):
                    print("Skipping grasp due to failed articulation")
                    pbar.set_description(
                        f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                    )
                    continue

                # Continue with regular simulation
                for step in range(500):
                    mujoco.mj_step(model, data)
                    if step % 20 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            return (
                                successful_transforms,
                                successful_qualities,
                                successful_widths,
                            )

                for step in range(approach_steps):
                    data.ctrl[2] = velocity
                    mujoco.mj_step(model, data)

                    current_pos = data.body("gripper_base").xpos
                    distance_to_target = np.linalg.norm(current_pos - pos)
                    if distance_to_target < 0.001:
                        break

                    if step % 50 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            return (
                                successful_transforms,
                                successful_qualities,
                                successful_widths,
                            )

                data.ctrl[1] = 0.0
                data.ctrl[2] = 0.0
                data.ctrl[3] = 0.0

                for step in range(100):
                    mujoco.mj_step(model, data)
                    if step % 20 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            return (
                                successful_transforms,
                                successful_qualities,
                                successful_widths,
                            )

                data.ctrl[0] = -1.0

                for step in range(1000):
                    mujoco.mj_step(model, data)
                    if step % 50 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            return (
                                successful_transforms,
                                successful_qualities,
                                successful_widths,
                            )

                for step in range(2000):
                    mujoco.mj_step(model, data)
                    if step % 20 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            return (
                                successful_transforms,
                                successful_qualities,
                                successful_widths,
                            )

                if not check_grasp(model, data, object_name, store_initial=True):
                    pbar.set_description(
                        f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                    )
                    continue

                directions = ["x", "y", "z"]
                shake_success = True

                for direction_idx, direction in enumerate(directions):
                    ctrl_idx = direction_idx + 1

                    for shake in range(args.num_shakes):
                        total_steps = args.shake_steps * 2

                        for step in range(total_steps):
                            angle = 2 * np.pi * step / total_steps
                            position = args.shake_magnitude * np.sin(angle)
                            data.ctrl[ctrl_idx] = position

                            mujoco.mj_step(model, data)
                            if step % 5 == 0:
                                viewer.sync()
                                if not viewer.is_running():
                                    return (
                                        successful_transforms,
                                        successful_qualities,
                                        successful_widths,
                                    )

                            if step == total_steps // 4 or step == 3 * total_steps // 4:
                                grasp_maintained = check_grasp(model, data, object_name)
                                if not grasp_maintained:
                                    shake_success = False
                                    break

                        if not shake_success:
                            break

                        data.ctrl[ctrl_idx] = 0
                        for step in range(50):
                            mujoco.mj_step(model, data)
                            if step % 20 == 0:
                                viewer.sync()
                                if not viewer.is_running():
                                    return (
                                        successful_transforms,
                                        successful_qualities,
                                        successful_widths,
                                    )

                    data.ctrl[ctrl_idx] = 0

                    if not shake_success:
                        break

                final_grasp_check = is_object_grasped(model, data, object_name)

                if shake_success and final_grasp_check:
                    successful_transforms.append(transform.tolist())
                    successful_qualities.append(quality)
                    successful_widths.append(width[i])

                    if (
                        args.max_successful > 0
                        and len(successful_transforms) >= args.max_successful
                    ):
                        tqdm.write(
                            f"Found {len(successful_transforms)} successful grasps (reached max_successful limit)"
                        )
                        return (
                            successful_transforms,
                            successful_qualities,
                            successful_widths,
                        )

                pbar.set_description(
                    f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                )

                for _ in range(100):
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    if not viewer.is_running():
                        return (
                            successful_transforms,
                            successful_qualities,
                            successful_widths,
                        )

        return successful_transforms, successful_qualities, successful_widths

    else:
        config = {
            "num_shakes": args.num_shakes,
            "shake_magnitude": args.shake_magnitude,
            "shake_steps": args.shake_steps,
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
                            (i, transform_result, quality_result, width[i])
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

            pbar = tqdm(
                total=len(grasp_params), desc=f"Testing grasps (0/0 successful)"
            )

            with mp.Pool(processes=num_workers) as pool:
                results = [
                    pool.apply_async(
                        test_single_grasp,
                        args=(param, object_name),
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

                    time.sleep(0.1)

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


if __name__ == "__main__":
    xml_path = os.path.join(os.path.dirname(__file__), "../assets/scene.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_name = args.object_name

    # Load and modify the XML file to remove free joints
    try:
        with open(args.xml_file, "r") as f:
            obj_xml_content = f.read()

        # Parse the XML
        obj_tree = ET.fromstring(obj_xml_content)

        # Find and remove all joints with type="free"
        free_joints = obj_tree.findall(".//joint[@type='free']")
        if free_joints:
            print(f"Found {len(free_joints)} free joints to remove")
            for joint in free_joints:
                # Find the parent of the joint directly
                for parent in obj_tree.findall(".//*"):
                    for child in parent.findall("joint"):
                        if (
                            child.get("name") == joint.get("name")
                            and child.get("type") == "free"
                        ):
                            parent.remove(child)
                            print(f"Removed free joint: {joint.get('name')}")

        # Update the original file with the modifications
        with open(args.xml_file, "w") as f:
            f.write(ET.tostring(obj_tree, encoding="unicode"))

        # Create include element using the updated file
        include = ET.Element("include", {"file": args.xml_file})
    except Exception as e:
        print(f"Error modifying XML to remove free joints: {e}")
        # Fall back to using original file if modification fails
        include = ET.Element("include", {"file": args.xml_file})

    root.append(include)

    worldbody = root.find("worldbody")

    # Add joint axis visualization cylinder
    # Try to load joint axis information from the analysis file
    joint_axis_file = args.grasps_path.replace("_grasps.json", "_joint_axis.json")
    if os.path.exists(joint_axis_file):
        try:
            with open(joint_axis_file, "r") as f:
                joint_data = json.load(f)

            primary_joint = joint_data.get("primary_joint")
            if primary_joint:
                # Use parent_position and parent_rotation from the joint axis analysis
                if (
                    "parent_position" in primary_joint
                    and "parent_rotation" in primary_joint
                ):
                    parent_pos = primary_joint["parent_position"]
                    parent_rot = primary_joint["parent_rotation"]

                    # Get global position
                    global_pos = [parent_pos["x"], parent_pos["y"], parent_pos["z"]]

                    # Format position for MuJoCo
                    cylinder_pos = f"{global_pos[0]} {global_pos[1]} {global_pos[2]}"

                    # Get the rotation axis for display
                    rotation_axis = primary_joint.get(
                        "rotation_axis", {"x": 0, "y": 1, "z": 0}
                    )
                    global_axis = [
                        rotation_axis["x"],
                        rotation_axis["y"],
                        rotation_axis["z"],
                    ]

                    # Calculate quaternion to align cylinder with the joint axis
                    # MuJoCo cylinders are aligned with Z-axis by default
                    default_axis = np.array([0, 0, 1])
                    joint_axis_vec = np.array(global_axis)

                    # Normalize the joint axis
                    joint_axis_norm = np.linalg.norm(joint_axis_vec)
                    if joint_axis_norm > 0:
                        joint_axis_vec = joint_axis_vec / joint_axis_norm

                    # Calculate rotation to align cylinder Z-axis with joint axis
                    if np.allclose(default_axis, joint_axis_vec):
                        # Already aligned
                        quat = [1, 0, 0, 0]  # Identity quaternion
                    elif np.allclose(default_axis, -joint_axis_vec):
                        # Opposite direction, rotate 180 degrees around X or Y
                        quat = [0, 1, 0, 0]  # 180 degrees around X
                    else:
                        # General case: use rotation between vectors
                        from scipy.spatial.transform import Rotation as R_scipy

                        # Create rotation that aligns default_axis with joint_axis_vec
                        cross_product = np.cross(default_axis, joint_axis_vec)
                        dot_product = np.dot(default_axis, joint_axis_vec)

                        # Handle the rotation
                        if np.linalg.norm(cross_product) > 1e-6:
                            # Normal case
                            rotation = R_scipy.align_vectors(
                                [joint_axis_vec], [default_axis]
                            )[0]
                            quat = rotation.as_quat(
                                scalar_first=True
                            )  # [w, x, y, z] format
                        else:
                            quat = [1, 0, 0, 0]  # Identity if vectors are parallel

                    quat_str = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"

                    # Add the joint axis cylinder (100m long)
                    joint_cylinder = ET.Element(
                        "geom",
                        {
                            "name": "joint_axis_cylinder",
                            "type": "cylinder",
                            "size": "0.005 50",  # radius 0.5cm, half-length 50m (total 100m)
                            "rgba": "1 1 0 0.5",  # Semi-transparent yellow
                            "pos": cylinder_pos,
                            "quat": quat_str,
                            "contype": "0",
                            "conaffinity": "0",
                        },
                    )
                    worldbody.append(joint_cylinder)

                    # Add a small sphere at the joint position for better visibility
                    joint_sphere = ET.Element(
                        "geom",
                        {
                            "name": "joint_position_marker",
                            "type": "sphere",
                            "size": "0.02",  # 2cm radius
                            "rgba": "1 1 0 0.8",  # Yellow
                            "pos": cylinder_pos,
                            "contype": "0",
                            "conaffinity": "0",
                        },
                    )
                    worldbody.append(joint_sphere)

                    # Store joint axis info for grasp-to-axis visualization
                    joint_axis_info = {
                        "position": np.array(global_pos),
                        "axis": np.array(global_axis),
                        "axis_normalized": joint_axis_vec,
                    }

                    print(f"Added joint axis visualization:")
                    print(f"  Joint: {primary_joint['name']}")
                    print(
                        f"  Global Position: [{global_pos[0]:.3f}, {global_pos[1]:.3f}, {global_pos[2]:.3f}]"
                    )
                    print(
                        f"  Global Axis: [{global_axis[0]:.3f}, {global_axis[1]:.3f}, {global_axis[2]:.3f}]"
                    )
                    print(
                        f"  Quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]"
                    )
                    print(f"  Joint Type: {primary_joint.get('type', 'unknown')}")
                    print(f"  Joint Range: {primary_joint.get('range', 'unlimited')}")

                    # Store joint axis info for grasp-to-axis visualization
                    joint_axis_info = {
                        "position": np.array(global_pos),
                        "axis": np.array(global_axis),
                        "axis_normalized": joint_axis_vec,
                    }

                else:
                    joint_axis_info = None
                    print(
                        "Warning: parent_position or parent_rotation not found in joint data"
                    )
                    print("Available fields:", list(primary_joint.keys()))

                # Print additional debugging information
                print(f"  Joint Type: {primary_joint.get('type', 'unknown')}")
                print(f"  Joint Range: {primary_joint.get('range', 'unlimited')}")
                if "body_hierarchy" in primary_joint:
                    print(f"  Body Hierarchy: {primary_joint['body_hierarchy']}")

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

    # Add grasp-to-axis perpendicular cylinders if we have joint axis info
    if joint_axis_info is not None:
        # Load grasp data to get grasp centers
        try:
            with open(args.grasps_path, "r") as f:
                grasp_data = json.load(f)
            transforms = grasp_data.get("transforms", [])

            print(
                f"\nAdding grasp-to-axis perpendicular visualization for {len(transforms)} grasps..."
            )

            for i, transform in enumerate(
                transforms[:1]
            ):  # Limit to first 20 grasps to avoid clutter
                # Extract grasp center position
                grasp_pos = np.array(transform)[:3, 3]

                axis_pos = joint_axis_info["position"]
                axis_dir = joint_axis_info["axis_normalized"]

                # Vector from axis position to grasp position
                to_grasp = grasp_pos - axis_pos

                # Project onto axis direction to find closest point parameter
                t = np.dot(to_grasp, axis_dir)

                # Closest point on axis
                closest_on_axis = axis_pos + t * axis_dir

                # Vector from closest point on axis to grasp center (perpendicular)
                perp_vector = grasp_pos - closest_on_axis
                perp_distance = np.linalg.norm(perp_vector)

                # Skip if distance is too small (grasp is very close to axis)
                if perp_distance < 0.001:
                    continue

                # Calculate midpoint for cylinder position
                cylinder_center = (grasp_pos + closest_on_axis) / 2

                # Calculate orientation to align cylinder with perpendicular vector
                perp_normalized = perp_vector / perp_distance

                # Align cylinder (default Z-axis) with perpendicular direction
                default_axis = np.array([0, 0, 1])
                if np.allclose(default_axis, perp_normalized):
                    perp_quat = [1, 0, 0, 0]  # Identity
                elif np.allclose(default_axis, -perp_normalized):
                    perp_quat = [0, 1, 0, 0]  # 180 degrees around X
                else:
                    from scipy.spatial.transform import Rotation as R_scipy

                    rotation = R_scipy.align_vectors([perp_normalized], [default_axis])[
                        0
                    ]
                    perp_quat = rotation.as_quat(scalar_first=True)  # [w, x, y, z]

                # Format for MuJoCo
                cylinder_pos_str = (
                    f"{cylinder_center[0]} {cylinder_center[1]} {cylinder_center[2]}"
                )
                quat_str = (
                    f"{perp_quat[0]} {perp_quat[1]} {perp_quat[2]} {perp_quat[3]}"
                )

                # Add perpendicular cylinder (green) - make it very long for visibility
                # Extend the cylinder much further in both directions
                extended_length = max(perp_distance * 5, 2.0)  # At least 2 meters long

                perp_cylinder = ET.Element(
                    "geom",
                    {
                        "name": f"grasp_to_axis_{i}",
                        "type": "cylinder",
                        "size": f"0.003 {extended_length / 2}",  # radius 0.3cm, half-length for extended cylinder
                        "rgba": "0 1 0 0.4",  # Semi-transparent green
                        "pos": cylinder_pos_str,
                        "quat": quat_str,
                        "contype": "0",
                        "conaffinity": "0",
                    },
                )
                worldbody.append(perp_cylinder)

                # Add small sphere at grasp center
                grasp_sphere = ET.Element(
                    "geom",
                    {
                        "name": f"grasp_center_{i}",
                        "type": "sphere",
                        "size": "0.005",  # 0.5cm radius
                        "rgba": "0 1 0 0.8",  # Green
                        "pos": f"{grasp_pos[0]} {grasp_pos[1]} {grasp_pos[2]}",
                        "contype": "0",
                        "conaffinity": "0",
                    },
                )
                worldbody.append(grasp_sphere)

                # Add small sphere at closest point on axis
                axis_point_sphere = ET.Element(
                    "geom",
                    {
                        "name": f"axis_point_{i}",
                        "type": "sphere",
                        "size": "0.003",  # 0.3cm radius
                        "rgba": "1 1 0 0.6",  # Yellow
                        "pos": f"{closest_on_axis[0]} {closest_on_axis[1]} {closest_on_axis[2]}",
                        "contype": "0",
                        "conaffinity": "0",
                    },
                )
                worldbody.append(axis_point_sphere)

                # Add circular trajectory visualization showing how the grasp moves during joint rotation
                circle_center = closest_on_axis  # Center of rotation on the joint axis
                radius = perp_distance  # Distance from joint axis to grasp center

                # Create a circle in 3D space around the joint axis
                # We need to create a coordinate system where the joint axis is one axis
                joint_axis = axis_dir  # Already normalized

                # Find two perpendicular vectors to the joint axis to define the plane of rotation
                # Start with an arbitrary vector and use Gram-Schmidt to get perpendicular vectors
                if abs(joint_axis[0]) < 0.9:
                    arbitrary = np.array([1, 0, 0])
                else:
                    arbitrary = np.array([0, 1, 0])

                # First perpendicular vector (in the plane of rotation)
                perp1 = arbitrary - np.dot(arbitrary, joint_axis) * joint_axis
                perp1 = perp1 / np.linalg.norm(perp1)

                # Second perpendicular vector (in the plane of rotation)
                perp2 = np.cross(joint_axis, perp1)
                perp2 = perp2 / np.linalg.norm(perp2)

                # Find the angle of the current grasp position relative to perp1
                current_vector = (
                    perp_vector / perp_distance
                )  # Normalized vector from axis to grasp
                current_angle = np.arctan2(
                    np.dot(current_vector, perp2), np.dot(current_vector, perp1)
                )

                # Get joint range from the primary joint data
                joint_range_str = primary_joint.get("range", "0 0")
                try:
                    range_parts = joint_range_str.split()
                    if len(range_parts) >= 2:
                        min_angle = float(range_parts[0])
                        max_angle = float(range_parts[1])
                    else:
                        # Default to full circle if range parsing fails
                        min_angle = 0
                        max_angle = 2 * np.pi
                except (ValueError, AttributeError):
                    # Default to full circle if range parsing fails
                    min_angle = 0
                    max_angle = 2 * np.pi

                # Calculate the angle range for trajectory
                angle_range = max_angle - min_angle

                # Create trajectory points only within the joint's valid range
                num_trajectory_points = max(
                    12, int(24 * angle_range / (2 * np.pi))
                )  # Scale points with range
                for j in range(num_trajectory_points):
                    # Angle for this trajectory point (within joint range)
                    # Start from current position and span the joint range
                    progress = (
                        j / (num_trajectory_points - 1)
                        if num_trajectory_points > 1
                        else 0
                    )
                    angle = current_angle + min_angle + (progress * angle_range)

                    # Calculate 3D position on the circle
                    circle_point = (
                        circle_center
                        + radius * np.cos(angle) * perp1
                        + radius * np.sin(angle) * perp2
                    )

                    # Color spheres differently based on position in range
                    if j == 0:
                        # First sphere (start position) - bright red
                        color = "1 0 0 0.9"
                    elif j == num_trajectory_points - 1:
                        # Last sphere (end position) - dark red
                        color = "0.5 0 0 0.9"
                    else:
                        # Middle spheres - gradient from red to orange
                        progress_color = progress
                        color = f"{1.0} {progress_color * 0.5} 0 0.7"

                    # Add trajectory sphere
                    trajectory_sphere = ET.Element(
                        "geom",
                        {
                            "name": f"trajectory_{i}_{j}",
                            "type": "sphere",
                            "size": "0.008",  # 0.8cm radius
                            "rgba": color,
                            "pos": f"{circle_point[0]} {circle_point[1]} {circle_point[2]}",
                            "contype": "0",
                            "conaffinity": "0",
                        },
                    )
                    worldbody.append(trajectory_sphere)

                print(
                    f"  Grasp {i}: Added trajectory with {num_trajectory_points} points"
                )
                print(
                    f"    Center: [{circle_center[0]:.3f}, {circle_center[1]:.3f}, {circle_center[2]:.3f}]"
                )
                print(f"    Radius: {radius:.3f}m")
                print(
                    f"    Joint range: {min_angle:.3f} to {max_angle:.3f} rad ({np.degrees(min_angle):.1f}° to {np.degrees(max_angle):.1f}°)"
                )
                print(f"    Current grasp angle: {np.degrees(current_angle):.1f}°")

            print(
                f"Added perpendicular cylinders for {min(len(transforms), 20)} grasps"
            )

        except Exception as e:
            print(f"Warning: Could not add grasp-to-axis visualization: {e}")
    else:
        print("No joint axis info available for grasp-to-axis visualization")

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
