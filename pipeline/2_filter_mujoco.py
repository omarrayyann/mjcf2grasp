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
    i, transform, quality, config = grasp_data

    xml_path = os.path.join(os.path.dirname(__file__), "../assets/scene.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    include = ET.Element("include", {"file": args.xml_file})
    root.append(include)

    xml_content = ET.tostring(root, encoding="unicode")
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


def run_simulation_with_viewer(model, data, object_name, use_viewer):
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
                mujoco.mj_resetData(model, data)

                pos = transform[:3, 3]
                quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

                geom_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, "test_sphere"
                )
                model.geom_pos[geom_id] = pos

                approach_distance = args.approach_distance
                approach_vector = transform[:3, 2] * approach_distance
                approach_pos = pos - approach_vector

                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base"
                )
                model.body_pos[body_id] = approach_pos
                model.body_quat[body_id] = quat

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
    xml_path = os.path.join(os.path.dirname(__file__), "../assets/scene.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_name = args.object_name

    include = ET.Element("include", {"file": args.xml_file})
    root.append(include)
    xml_content = ET.tostring(root, encoding="unicode")

    gripper_xml_path = os.path.join(
        os.path.dirname(__file__), "../assets/gripper_models/rum_gripper/model.xml"
    )
    with open(gripper_xml_path, "r") as f:
        additional_xml_content = f.read()
    xml_content = merge_xml_contents(xml_content, additional_xml_content)

    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    successful_transforms, successful_qualities, successful_widths = (
        run_simulation_with_viewer(model, data, object_name, args.render)
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
