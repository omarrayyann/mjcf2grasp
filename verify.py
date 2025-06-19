import time
import numpy as np
import mujoco
import mujoco.viewer
import os
import xml.etree.ElementTree as ET
import argparse
import json
from scipy.spatial.transform import Rotation as R

initial_relative_position = None
initial_grasp_verified = False


def list_all_geoms(model):
    """List all geometry names in the model to help identify finger geoms"""
    print("\n=== Available Geometry Names ===")
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        print(f"  Geom {i}: {geom_name}")
    print("================================\n")


def is_object_grasped(model, data, verbose=False):
    """
    An object is considered "picked up" if it's in contact with both fingers.
    This function dynamically identifies finger geoms based on common naming patterns.
    """
    left_finger_contact = False
    right_finger_contact = False
    contact_count = 0

    # List to store contacts for debugging
    contacts_list = []

    # Possible finger name patterns
    left_patterns = ["left_finger", "finger_l", "gripper_finger_left"]
    right_patterns = ["right_finger", "finger_r", "gripper_finger_right"]

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

        # Skip if either name is None
        if not geom1_name or not geom2_name:
            continue

        # Check for object contact
        if geom1_name == "object" or geom2_name == "object":
            other_geom = geom2_name if geom1_name == "object" else geom1_name
            contacts_list.append(other_geom)

            # Check if the other geom is a left finger
            for pattern in left_patterns:
                if pattern in str(other_geom).lower():
                    left_finger_contact = True
                    contact_count += 1
                    if verbose:
                        print(f"Left finger contact: {other_geom}")
                    break

            # Check if the other geom is a right finger
            for pattern in right_patterns:
                if pattern in str(other_geom).lower():
                    right_finger_contact = True
                    contact_count += 1
                    if verbose:
                        print(f"Right finger contact: {other_geom}")
                    break

    if verbose:
        print(f"Total object contacts found: {contact_count}")
        print(f"All contacts with object: {contacts_list}")
        print(f"Left finger contact: {left_finger_contact}")
        print(f"Right finger contact: {right_finger_contact}")

        # If we're missing a contact, suggest checking the scene for finger names
        if not (left_finger_contact and right_finger_contact):
            print("Warning: Not all finger contacts detected!")
            print(
                "Check the finger geometry names in the scene and update the patterns"
            )
            print(f"Current left patterns: {left_patterns}")
            print(f"Current right patterns: {right_patterns}")

    # Both fingers must be in contact with the object for a successful grasp
    return left_finger_contact and right_finger_contact


def check_grasp(model, data, store_initial=False, verbose=False):
    global initial_relative_position, initial_grasp_verified

    object_pos = data.body("object_body").xpos
    gripper_pos = data.body("gripper_base").xpos

    relative_position = object_pos - gripper_pos

    if store_initial:
        # Verify that the object is actually being grasped AFTER closing the gripper
        if is_object_grasped(model, data, verbose=verbose):
            # Store the object position relative to the gripper after successful grasp
            initial_relative_position = relative_position.copy()
            initial_grasp_verified = True
            if verbose:
                print("Initial grasp verified (post-grasp position).")
                print(f"Object position: {object_pos}")
                print(f"Gripper position: {gripper_pos}")
                print(
                    f"Relative position (object-gripper): {initial_relative_position}"
                )
            return True
        else:
            if verbose:
                print(
                    "Initial grasp verification failed: Object not properly grasped by both fingers"
                )
            initial_grasp_verified = False
            return False

    # If initial position wasn't verified, grasp check fails
    if not initial_grasp_verified or initial_relative_position is None:
        if verbose:
            print("Grasp check failed: Initial position not verified")
        return False

    position_change = np.linalg.norm(relative_position - initial_relative_position)
    max_allowed_change = 0.05

    # Make sure the object is still in contact with the gripper
    still_grasped = is_object_grasped(model, data, verbose=verbose)

    if verbose:
        print(f"Position change: {position_change}, Max allowed: {max_allowed_change}")
        print(f"Still in contact: {still_grasped}")

    print("position_change:", position_change)
    return position_change < max_allowed_change and still_grasped


parser = argparse.ArgumentParser()
parser.add_argument("object", type=str)
parser.add_argument("--num_shakes", type=int, default=5)
parser.add_argument("--shake_magnitude", type=float, default=0.01)
parser.add_argument("--top_k", type=int, default=10)
args = parser.parse_args()

mesh_file = os.path.abspath(f"assets/objects/{args.object}.obj")
mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
object_name = os.path.splitext(os.path.basename(mesh_file))[0]
xml_path = os.path.join(os.path.dirname(__file__), "assets/scene.xml")
tree = ET.parse(xml_path)
root = tree.getroot()
asset = root.find("asset")
if asset is not None:
    mesh = ET.Element("mesh", {"file": mesh_file, "scale": "1 1 1"})
    asset.append(mesh)
worldbody = root.find("worldbody")
if worldbody is not None:
    body = ET.Element("body", {"name": "object_body", "gravcomp": "1"})
    joint = ET.Element("joint", {"type": "free", "damping": "10."})
    geom = ET.Element(
        "geom",
        {
            "name": "object",
            "type": "mesh",
            "mesh": object_name,
        },
    )
    body.append(joint)
    body.append(geom)
    worldbody.append(body)

modified_xml_content = ET.tostring(root, encoding="unicode")

json_file_path = f"output/{object_name}_grasps.json"
with open(json_file_path, "r") as f:
    data = json.load(f)
transforms = np.array(data["transforms"])
quality = np.array(
    data.get(
        "quality_antipodal",
        data.get("quality_number_of_contacts", [1.0] * len(transforms)),
    )
)
top_k = args.top_k
top_indices = np.argsort(quality)[-top_k:][::-1]
transforms = [transforms[i] for i in top_indices]
qualities = [quality[i] for i in top_indices]

grasp_results = []

model = mujoco.MjModel.from_xml_string(modified_xml_content)
data = mujoco.MjData(model)

# List all geometry names to help identify finger geoms
list_all_geoms(model)

with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:
    for i, (transform, quality) in enumerate(zip(transforms, qualities)):
        print(f"Testing grasp {i + 1}/{len(transforms)} (quality: {quality:.4f})")

        mujoco.mj_resetData(model, data)

        gripper_pos = transform[:3, 3]
        gripper_rot = transform[:3, :3]
        gripper_quat = R.from_matrix(gripper_rot).as_quat(scalar_first=True)

        model.body_pos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        ] = gripper_pos
        model.body_quat[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        ] = gripper_quat

        mujoco.mj_step(model, data)

        # Close gripper to establish grasp
        data.ctrl[0] = -1.0
        for _ in range(2000):
            mujoco.mj_step(model, data)
            if _ % 100 == 0:
                viewer.sync()
                if not viewer.is_running():
                    break

        # After closing the gripper, wait for stabilization
        print("Waiting for grasp stabilization...")
        for _ in range(500):
            mujoco.mj_step(model, data)
            if _ % 100 == 0:
                viewer.sync()
                if not viewer.is_running():
                    break

        # Now verify the object is actually grasped and store initial position post-grasp
        print(f"\nVerifying initial grasp for grasp {i + 1}...")
        grasp_successful = check_grasp(model, data, store_initial=True, verbose=True)

        if not grasp_successful:
            print(f"Grasp {i + 1} failed initial check - Object not grasped")
            grasp_results.append(
                {
                    "grasp_index": i,
                    "quality": quality,
                    "success": False,
                    "failed_at": "initial_no_contact",
                }
            )
            time.sleep(5)
            continue

        directions = ["x", "y", "z"]
        shake_success = True
        failed_direction = None

        original_pos = model.body_pos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        ].copy()

        for direction_idx, direction in enumerate(directions):
            print(f"  Shaking in {direction} direction...")

            ctrl_idx = direction_idx + 1

            for shake in range(args.num_shakes):
                data.ctrl[ctrl_idx] = args.shake_magnitude

                for _ in range(500):
                    mujoco.mj_step(model, data)
                    if _ % 50 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            break

                # Check if grasp is maintained during positive shake
                print(f"\nChecking grasp during {direction}+ shake {shake + 1}...")
                grasp_maintained = check_grasp(model, data, verbose=True)
                if not grasp_maintained:
                    print(
                        f"  Grasp {i + 1} failed during {direction}+ shake {shake + 1}"
                    )
                    shake_success = False
                    failed_direction = f"{direction}+"
                    break

                data.ctrl[ctrl_idx] = -args.shake_magnitude

                for _ in range(500):
                    mujoco.mj_step(model, data)
                    if _ % 50 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            break

                # Check if grasp is maintained during negative shake
                print(f"\nChecking grasp during {direction}- shake {shake + 1}...")
                grasp_maintained = check_grasp(model, data, verbose=True)
                if not grasp_maintained:
                    print(
                        f"  Grasp {i + 1} failed during {direction}- shake {shake + 1}"
                    )
                    shake_success = False
                    failed_direction = f"{direction}-"
                    break

                data.ctrl[ctrl_idx] = 0

                for _ in range(200):
                    mujoco.mj_step(model, data)
                    if _ % 50 == 0:
                        viewer.sync()
                        if not viewer.is_running():
                            break

            data.ctrl[ctrl_idx] = 0

            if not shake_success:
                break

        # Final check to verify two-finger grasp is still maintained
        final_grasp_check = is_object_grasped(model, data, verbose=True)

        grasp_results.append(
            {
                "grasp_index": i,
                "quality": quality,
                "success": shake_success and final_grasp_check,
                "failed_at": failed_direction if not shake_success else None,
                "final_two_finger_grasp": final_grasp_check,
            }
        )

        print(f"\nGrasp {i + 1} Summary:")
        print(f"  Shake tests: {'PASSED' if shake_success else 'FAILED'}")
        print(f"  Two-finger grasp: {'MAINTAINED' if final_grasp_check else 'LOST'}")
        print(
            f"  Overall: {'SUCCEEDED' if (shake_success and final_grasp_check) else 'FAILED'}"
        )

        time.sleep(1)

        if not viewer.is_running():
            break

results_file = f"output/{object_name}_verification_results.json"
with open(results_file, "w") as f:
    json.dump({"results": grasp_results}, f, indent=2)

print(f"Results saved to {results_file}")
