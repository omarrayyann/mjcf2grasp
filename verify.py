import time
import numpy as np
import mujoco
import mujoco.viewer
import os
import xml.etree.ElementTree as ET
import argparse
import json
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("object", type=str)
parser.add_argument("--num_shakes", type=int, default=2)
parser.add_argument("--shake_magnitude", type=float, default=0.05)
parser.add_argument("--shake_steps", type=int, default=100)
parser.add_argument("--render", action="store_true", 
                    help="Enable interactive viewer")
args = parser.parse_args()

initial_relative_position = None
initial_grasp_verified = False


def is_object_grasped(model, data, verbose=False):
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

        if "object" in geom1 or "object" in geom2:
            other = geom2 if "object" in geom1 else geom1
            if any(p in other.lower() for p in left_patterns):
                left_finger_contact = True
            if any(p in other.lower() for p in right_patterns):
                right_finger_contact = True

    return left_finger_contact and right_finger_contact


def check_grasp(model, data, store_initial=False, verbose=False):
    global initial_relative_position, initial_grasp_verified

    object_pos = data.body("object_body").xpos
    gripper_pos = data.site("end_effector").xpos
    relative_position = object_pos - gripper_pos

    if store_initial:
        if is_object_grasped(model, data):
            initial_relative_position = relative_position.copy()
            initial_grasp_verified = True
            return True
        initial_grasp_verified = False
        return False

    if not initial_grasp_verified or initial_relative_position is None:
        return False

    position_change = np.linalg.norm(relative_position - initial_relative_position)
    return position_change < 0.01 and is_object_grasped(model, data)


def run_simulation_with_viewer(model, data, use_viewer):
    """Run the simulation with or without interactive viewer"""
    
    # Load all grasps
    with open(f"output/{object_name}_grasps.json", "r") as f:
        grasp_data = json.load(f)
    transforms = np.array(grasp_data["transforms"])
    qualities = np.array(grasp_data.get("quality_antipodal", [1.0] * len(transforms)))

    # Grasp testing loop
    successful_transforms = []
    successful_qualities = []

    def test_grasps():
        nonlocal successful_transforms, successful_qualities
        
        for i, (transform, quality) in enumerate(zip(transforms, qualities)):
            if len(successful_transforms) >= 50:
                break

            mujoco.mj_resetData(model, data)

            pos = transform[:3, 3]
            quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
            model.body_pos[body_id] = pos
            model.body_quat[body_id] = quat

            mujoco.mj_step(model, data)
            data.ctrl[0] = -1.0

            # Close gripper
            for step in range(1000):
                mujoco.mj_step(model, data)
                if use_viewer and step % 50 == 0:
                    viewer.sync()
                    if not viewer.is_running():
                        return

            # Stabilize grasp
            for step in range(200):
                mujoco.mj_step(model, data)
                if use_viewer and step % 20 == 0:
                    viewer.sync()
                    if not viewer.is_running():
                        return

            if not check_grasp(model, data, store_initial=True):
                print(f"Grasp {i + 1} failed initial check")
                continue

            print(f"Testing grasp {i + 1}/{len(transforms)}...")
            
            shake_success = True
            for axis in range(3):
                ctrl_idx = axis + 1
                for shake_num in range(args.num_shakes):
                    # Positive shake
                    for step in range(args.shake_steps):
                        data.ctrl[ctrl_idx] = args.shake_magnitude * (step + 1) / args.shake_steps
                        mujoco.mj_step(model, data)
                        if use_viewer and step % 5 == 0:
                            viewer.sync()
                            if not viewer.is_running():
                                return

                    # Check grasp after positive shake
                    for step in range(50):
                        mujoco.mj_step(model, data)
                        if use_viewer and step % 10 == 0:
                            viewer.sync()
                            if not viewer.is_running():
                                return

                    if not check_grasp(model, data):
                        shake_success = False
                        break

                    # Negative shake
                    for step in range(args.shake_steps * 2):
                        offset = args.shake_magnitude * (1 - (step + 1) / args.shake_steps)
                        if step >= args.shake_steps:
                            offset = -args.shake_magnitude * ((step + 1 - args.shake_steps) / args.shake_steps)
                        data.ctrl[ctrl_idx] = offset
                        mujoco.mj_step(model, data)
                        if use_viewer and step % 5 == 0:
                            viewer.sync()
                            if not viewer.is_running():
                                return

                    # Check grasp after negative shake
                    for step in range(50):
                        mujoco.mj_step(model, data)
                        if use_viewer and step % 10 == 0:
                            viewer.sync()
                            if not viewer.is_running():
                                return

                    if not check_grasp(model, data):
                        shake_success = False
                        break

                data.ctrl[ctrl_idx] = 0
                if not shake_success:
                    break

            if shake_success and is_object_grasped(model, data):
                successful_transforms.append(transform.tolist())
                successful_qualities.append(quality)
                print(f"Grasp {i + 1} SUCCESSFUL ({len(successful_transforms)}/50)")
            else:
                print(f"Grasp {i + 1} failed")

            # Add delay for viewing
            if use_viewer:
                for _ in range(100):  # Brief pause between grasps
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    if not viewer.is_running():
                        return

    # Run simulation based on viewer mode
    if use_viewer:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
            test_grasps()
    else:
        test_grasps()

    return successful_transforms, successful_qualities


# Load object and scene
mesh_file = os.path.abspath(f"assets/objects/{args.object}.obj")
object_name = os.path.splitext(os.path.basename(mesh_file))[0]
xml_path = os.path.join(os.path.dirname(__file__), "assets/scene.xml")
tree = ET.parse(xml_path)
root = tree.getroot()

asset = root.find("asset")
mesh = ET.Element("mesh", {"file": mesh_file, "scale": "1 1 1"})
asset.append(mesh)

worldbody = root.find("worldbody")
body = ET.Element("body", {"name": "object_body", "gravcomp": "1"})
joint = ET.Element("joint", {"type": "free", "damping": "10."})
geom = ET.Element("geom", {"name": "object", "type": "mesh", "mesh": object_name})
body.extend([joint, geom])
worldbody.append(body)

xml_content = ET.tostring(root, encoding="unicode")
model = mujoco.MjModel.from_xml_string(xml_content)
data = mujoco.MjData(model)

print(f"Running simulation{'with interactive viewer' if args.render else ''}")

# Run the simulation
successful_transforms, successful_qualities = run_simulation_with_viewer(model, data, args.render)

# Save updated successful grasps
output_path = f"output/{object_name}_grasps_verified.json"
with open(output_path, "w") as f:
    # Load original grasp data to preserve other fields
    with open(f"output/{object_name}_grasps.json", "r") as original_f:
        original_data = json.load(original_f)
    
    json.dump({
        "transforms": successful_transforms,
        "quality_antipodal": successful_qualities,
        "object": original_data.get('object', object_name),
        "object_scale": original_data.get('object_scale', 1.0),
    }, f, indent=2)

print(f"\nSaved {len(successful_transforms)} successful grasps to {output_path}")