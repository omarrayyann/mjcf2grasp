import json
import os
import subprocess

# Load JSON data from file
with open("matched_objs.json", "r") as f:
    data = json.load(f)

# Base paths
thor_assets_path = "/home/lambda1/Documents/thor-grasp/assets/Thor-Assets"
base_input_path = "../../../assets/objects"
temp_folder = "tmp"
os.makedirs(temp_folder, exist_ok=True)
temp_path = os.path.join(temp_folder, "temp.obj")
temp_abs_path = os.path.abspath(temp_path)
output_path = os.path.join(temp_folder, "output.obj")
output_abs_path = os.path.abspath(output_path)

for obj in data:
    # Get full .obj path and derive object name (without extension)
    relative_path = obj["path"]
    full_path = os.path.join(thor_assets_path, relative_path)
    object_name = os.path.splitext(os.path.basename(full_path))[0]

    print(f"\nProcessing object: {object_name} manifold")
    # subprocess.run([
    #     "./manifold", full_path, temp_abs_path, "-s"
    # ], cwd="external_src/Manifold/build", check=True)

    # print(f"\nProcessing object: {object_name} simplification")

    # subprocess.run([
    #     "./simplify", "-i", temp_abs_path, "-o", output_abs_path, "-m", "-r", "0.02"
    # ], cwd="external_src/Manifold/build", check=True)

    # print(f"Generating grasps for object: {object_name}")
    # subprocess.run([
    #     "python", "pipeline/0_generate_grasps.py",
    #     "--object_file", output_abs_path,
    #     "--quality", "antipodal",
    #     "--output", f"output/{object_name}_grasps.json",
    #     "--systematic_sampling"
    # ], check=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "myenv/lib/python3.10/site-packages/PySide2/Qt/lib:" + env.get("LD_LIBRARY_PATH", "")

    print(f"Visualizing initial grasps for object: {object_name}")
    subprocess.run([
        "python", "scripts/visualize.py", object_name
    ], check=True, env=env)

    # print(f"Filtering grasps for object: {object_name} using MuJoCo")
    # subprocess.run([
    #     "python", "pipeline/1_filter_mujoco.py", "--mesh_path", output_abs_path, "--grasps_path", f"output/{object_name}_grasps.json"
    # ], check=True)

    print(f"Visualizing filtered grasps for object: {object_name}")
    subprocess.run([
        "python", "scripts/visualize.py", object_name, "--filtered"
    ], check=True)
