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
parser.add_argument("--mesh_path", type=str)
parser.add_argument("--grasps_path", type=str)
parser.add_argument("--num_shakes", type=int, default=2)
parser.add_argument("--shake_magnitude", type=float, default=0.1)
parser.add_argument("--shake_steps", type=int, default=100)
parser.add_argument("--render", action="store_true", 
                    help="Enable interactive viewer")
parser.add_argument("--num_workers", type=int, default=mp.cpu_count(),
                    help="Number of parallel processes to use")
args = parser.parse_args()

initial_relative_position = None
initial_grasp_verified = False


def is_object_grasped(model, data):
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


def check_grasp(model, data, store_initial=False):
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
    return position_change < 0.03 and is_object_grasped(model, data)


def test_single_grasp(grasp_data):
    """Test a single grasp in a separate process"""
    i, transform, quality, config = grasp_data
    
    # Create a new MuJoCo model and data for this process
    mesh_file = os.path.abspath(config['mesh_path'])
    xml_path = os.path.join(os.path.dirname(__file__), "../assets/scene.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    asset = root.find("asset")
    mesh = ET.Element("mesh", {"file": mesh_file, "scale": "1 1 1", "name":"object_mesh"})
    asset.append(mesh)

    worldbody = root.find("worldbody")
    body = ET.Element("body", {"name": "object_body", "gravcomp": "1"})
    joint = ET.Element("joint", {"type": "free", "damping": "10."})
    geom = ET.Element("geom", {"name": "object", "type": "mesh", "mesh": "object_mesh"})
    body.extend([joint, geom])
    worldbody.append(body)

    xml_content = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    
    # Reset variables for this process
    global initial_relative_position, initial_grasp_verified
    initial_relative_position = None
    initial_grasp_verified = False
    
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

    # Stabilize grasp
    for step in range(2000):
        mujoco.mj_step(model, data)

    if not check_grasp(model, data, store_initial=True):
        return i, None, None  # Failed grasp
    
    # Updated shaking logic
    directions = ["x", "y", "z"]
    shake_success = True
    
    for direction_idx, direction in enumerate(directions):
        ctrl_idx = direction_idx + 1

        for shake in range(config['num_shakes']):
            total_steps = config['shake_steps'] * 2

            for step in range(total_steps):
                # Sinusoidal motion
                angle = 2 * np.pi * step / total_steps
                position = config['shake_magnitude'] * np.sin(angle)
                data.ctrl[ctrl_idx] = position

                mujoco.mj_step(model, data)

                # Check grasp at quarter and three-quarter points
                if step == total_steps // 4 or step == 3 * total_steps // 4:
                    grasp_maintained = check_grasp(model, data)
                    if not grasp_maintained:
                        shake_success = False
                        break

            if not shake_success:
                break

            # Reset control and stabilize
            data.ctrl[ctrl_idx] = 0
            for step in range(50):
                mujoco.mj_step(model, data)

        # Reset control for this direction
        data.ctrl[ctrl_idx] = 0

        if not shake_success:
            break

    # Final grasp check
    final_grasp_check = is_object_grasped(model, data)

    if shake_success and final_grasp_check:
        return i, transform.tolist(), quality
    else:
        return i, None, None


def run_simulation_with_viewer(model, data, use_viewer):
    """Run the simulation with or without interactive viewer"""
    
    # Load all grasps
    with open(args.grasps_path, "r") as f:
        grasp_data = json.load(f)
    transforms = np.array(grasp_data["transforms"])
    qualities = np.array(grasp_data.get("quality_antipodal", [1.0] * len(transforms)))

    # Handle viewer mode separately (can't parallelize with viewer)
    if use_viewer:
        successful_transforms = []
        successful_qualities = []
        
        with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
            pbar = tqdm(enumerate(zip(transforms, qualities)), 
                       total=len(transforms),
                       desc=f"Testing grasps (0/0 successful)")
            
            for i, (transform, quality) in pbar:
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
                    if step % 50 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            return successful_transforms, successful_qualities

                # Stabilize grasp
                for step in range(2000):
                    mujoco.mj_step(model, data)
                    if step % 20 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            return successful_transforms, successful_qualities

                if not check_grasp(model, data, store_initial=True):
                    pbar.set_description(f"Testing grasps ({len(successful_transforms)}/{i+1} successful)")
                    continue
                
                # Updated shaking logic
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
                                    return successful_transforms, successful_qualities

                            # Check grasp at quarter and three-quarter points
                            if step == total_steps // 4 or step == 3 * total_steps // 4:
                                grasp_maintained = check_grasp(model, data)
                                if not grasp_maintained:
                                    shake_success = False
                                    break

                        if not shake_success:
                            break

                        # Reset control and stabilize
                        data.ctrl[ctrl_idx] = 0
                        for step in range(50):
                            mujoco.mj_step(model, data)
                            if step % 20 == 0:
                                viewer.sync()
                                if not viewer.is_running():
                                    return successful_transforms, successful_qualities

                    # Reset control for this direction
                    data.ctrl[ctrl_idx] = 0

                    if not shake_success:
                        break

                # Final grasp check
                final_grasp_check = is_object_grasped(model, data)

                if shake_success and final_grasp_check:
                    successful_transforms.append(transform.tolist())
                    successful_qualities.append(quality)
                
                # Update progress bar with success rate
                pbar.set_description(f"Testing grasps ({len(successful_transforms)}/{i+1} successful)")

                # Add delay for viewing
                for _ in range(100):  # Brief pause between grasps
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    if not viewer.is_running():
                        return successful_transforms, successful_qualities
                        
        return successful_transforms, successful_qualities
    
    # Parallel processing for non-viewer mode
    else:
        # Prepare grasp data for parallel processing
        config = {
            'mesh_path': args.mesh_path,
            'num_shakes': args.num_shakes,
            'shake_magnitude': args.shake_magnitude,
            'shake_steps': args.shake_steps
        }
        
        grasp_params = [(i, transform, quality, config) 
                        for i, (transform, quality) in enumerate(zip(transforms, qualities))]
        
        # Create process pool
        num_workers = min(args.num_workers, len(grasp_params))
        
        successful_transforms = []
        successful_qualities = []
        
        # Use a manager to share progress information
        with mp.Manager() as manager:
            # Create shared values - Manager's Value objects don't use get_lock()
            success_count = manager.Value('i', 0)
            processed_count = manager.Value('i', 0)
            lock = manager.Lock()
            
            # Function to update the progress bar
            def update_progress_bar(result):
                nonlocal pbar
                i, transform_result, quality_result = result
                
                # Use the shared lock for atomic updates
                with lock:
                    processed_count.value += 1
                    
                    if transform_result is not None:
                        success_count.value += 1
                        successful_transforms.append((i, transform_result, quality_result))
                
                pbar.set_description(f"Testing grasps ({success_count.value}/{processed_count.value} successful)")
                pbar.update(1)
            
            # Create progress bar
            pbar = tqdm(total=len(grasp_params), desc=f"Testing grasps (0/0 successful)")
            
            # Start parallel processing
            with mp.Pool(processes=num_workers) as pool:
                results = [pool.apply_async(test_single_grasp, args=(param,), callback=update_progress_bar) 
                           for param in grasp_params]
                
                # Ensure all processes complete
                for result in results:
                    result.wait()
                
                # Close and join the pool
                pool.close()
                pool.join()
            
            # Sort results by original index
            successful_transforms.sort()
            # Extract just the transform and quality, discarding the index
            successful_transforms_only = [t for _, t, _ in successful_transforms]
            successful_qualities_only = [q for _, _, q in successful_transforms]
            
        pbar.close()
        return successful_transforms_only, successful_qualities_only


if __name__ == "__main__":
    # Only needed for the interactive viewer mode or as a baseline model for the parallel version
    mesh_file = os.path.abspath(args.mesh_path)
    xml_path = os.path.join(os.path.dirname(__file__), "../assets/scene.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    asset = root.find("asset")
    mesh = ET.Element("mesh", {"file": mesh_file, "scale": "1 1 1", "name":"object_mesh"})
    asset.append(mesh)

    worldbody = root.find("worldbody")
    body = ET.Element("body", {"name": "object_body", "gravcomp": "1"})
    joint = ET.Element("joint", {"type": "free", "damping": "10."})
    geom = ET.Element("geom", {"name": "object", "type": "mesh", "mesh": "object_mesh"})
    body.extend([joint, geom])
    worldbody.append(body)

    xml_content = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    # Run the simulation
    successful_transforms, successful_qualities = run_simulation_with_viewer(model, data, args.render)

    # Save updated successful grasps
    output_path = args.grasps_path.replace(".json", "_filtered.json")
    with open(output_path, "w") as f:
        # Load original grasp data to preserve other fields
        with open(args.grasps_path, "r") as original_f:
            original_data = json.load(original_f)
        
        json.dump({
            "transforms": successful_transforms,
            "quality_antipodal": successful_qualities,
            "object": original_data.get('object', "unknown_object"),
            "object_scale": original_data.get('object_scale', 1.0),
        }, f, indent=2)

    tqdm.write(f"Saved {len(successful_transforms)} successful grasps to {output_path}")
    
    if not args.render and args.num_workers > 1:
        tqdm.write(f"Used {args.num_workers} parallel workers for grasp testing")