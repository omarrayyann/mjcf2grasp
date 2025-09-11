import numpy as np
import json
import glob
import os
from scipy.spatial.transform import Rotation as R

OUTPUT_DIR = "output_articulate"


def transform_grasps(joint_name, joint_info, filtered_grasps_file):
    joint_parent_body = joint_info["parent_body"]
    joint_parent_position = [joint_info["parent_position"]["x"], joint_info["parent_position"]["y"], joint_info["parent_position"]["z"]]
    joint_parent_rotation = [joint_info["parent_rotation"]["x"], joint_info["parent_rotation"]["y"], joint_info["parent_rotation"]["z"], joint_info["parent_rotation"]["w"]]
    joint_parent_transform = np.eye(4)
    joint_parent_transform[:3, 3] = joint_parent_position
    joint_parent_transform[:3, :3] = R.from_quat(joint_parent_rotation).as_matrix()

    new_transforms = []
    with open(filtered_grasps_file, "r") as f:
        data = json.load(f)
        transforms = np.array(data.get("root_transforms", data["transforms"]))
        for transform in transforms:
            transform = np.array(transform)
            # grasp from base frame = transform
            # parent frame from base frame = joint_parent_transform
            # grasp from parent frame = inverse(parent from base) × (grasp from base)
            transform = np.linalg.inv(joint_parent_transform) @ transform 
            transform = transform.tolist()
            new_transforms.append(transform)
    data["root_transforms"] = transforms.tolist() # data["transforms"]
    data["parent_transform"] = new_transforms
    #data["transforms"] = new_transforms
    data["joint_info"] = joint_info
    with open(filtered_grasps_file, "w") as f:
        json.dump(data, f, indent=2)

def main():
    for file in glob.glob(f"{OUTPUT_DIR}/*/joint_meshes_info_filtered.json"):
        json_file = os.path.abspath(file)
        with open(json_file, "r") as f:
            data = json.load(f)
            for entry in data:
                joint_name = entry["joint"]
                joint_info = entry["joint_info"]
                filtered_grasps_file = entry["filtered_grasps_file"]
                if filtered_grasps_file is None:
                    continue
                print(json_file, filtered_grasps_file)
                abs_filtered_grasps_file = os.path.join(os.path.dirname(json_file), filtered_grasps_file)
                transform_grasps(joint_name, joint_info, abs_filtered_grasps_file)

if __name__ == "__main__":
    main()
