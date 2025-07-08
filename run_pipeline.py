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
    full_path = obj["path"]
    object_name = os.path.splitext(os.path.basename(full_path))[0]
    xml_file = obj["xml"]


    # print(f"\nProcessing object: {object_name} manifold")
    # subprocess.run([
    #     "./manifold", full_path, temp_abs_path, "-s"
    # ], cwd="external_src/Manifold/build", check=True)

    # print(f"\nProcessing object: {object_name} simplification")

    # subprocess.run([
    #     "./simplify", "-i", temp_abs_path, "-o", output_abs_path, "-m", "-r", "0.02"
    # ], cwd="external_src/Manifold/build", check=True)

    # print(f"Generating grasps for object: {object_name}")
    # try:
    #     subprocess.run([
    #         "python", "pipeline/0_generate_grasps.py",
    #         "--object_file", output_abs_path,
    #         "--quality", "antipodal",
    #         "--output", f"output/{object_name}_grasps.json",
    #         "--systematic_sampling",
    #         "--num_workers", str(os.cpu_count())  # Use all available CPU cores
    #     ], check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error generating grasps for {object_name}: {str(e)}")
    #     print("Trying again with fewer workers...")
    #     # Try again with half the workers
    #     num_workers = max(1, os.cpu_count() // 2)
    #     subprocess.run([
    #         "python", "pipeline/0_generate_grasps.py",
    #         "--object_file", output_abs_path,
    #         "--quality", "antipodal",
    #         "--output", f"output/{object_name}_grasps.json",
    #         "--systematic_sampling",
    #         "--num_workers", str(num_workers)
    #     ], check=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "myenv/lib/python3.10/site-packages/PySide2/Qt/lib:" + env.get("LD_LIBRARY_PATH", "")

    # print(f"Visualizing initial grasps for object: {object_name}")
    # subprocess.run([
    #     "python", "scripts/visualize.py", object_name
    # ], check=True, env=env)

    # print(f"Filtering grasps for object: {object_name} using MuJoCo")
    # try:
    #     subprocess.run([
    #         "python", "pipeline/1_filter_mujoco.py", 
    #         "--mesh_path", full_path, 
    #         "--grasps_path", f"output/{object_name}_grasps.json",
    #         "--xml_file", xml_file,
    #         "--num_workers", str(os.cpu_count()),
    #         "--approach_distance", "0.1",   # Start gripper 10cm away from grasp point
    #         "--approach_steps", "1000",    # Number of steps for approach
    #         # "--render",  # Enable rendering for visualization
    #         "--max_successful", "1000"  # Stop after finding 1000 successful grasps
    #     ], check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error filtering grasps for {object_name}: {str(e)}")
    #     print("Trying again with fewer workers...")
    #     # Try again with half the workers
    #     num_workers = max(1, os.cpu_count() // 2)
    #     subprocess.run([
    #         "python", "pipeline/1_filter_mujoco.py", 
    #         "--mesh_path", full_path, 
    #         "--grasps_path", f"output/{object_name}_grasps.json",
    #         "--xml_file", xml_file,
    #         "--num_workers", str(num_workers),
    #         "--approach_distance", "0.1",   # Start gripper 10cm away from grasp point
    #         "--approach_steps", "1000",    # Number of steps for approach
    #         # "--render",  # Enable rendering for visualization
    #         "--max_successful", "1000"  # Stop after finding 1000 successful grasps
    #     ], check=True)

    print(f"Visualizing filtered grasps for object: {object_name}")
    try:
        subprocess.run([
            "python", "scripts/visualize.py", object_name, "--filtered"
        ], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Visualization for {object_name} failed: {str(e)}")
        print("Continuing to next object...")
    
    # print(f"Visualizing comparison between filtered and unfiltered grasps for object: {object_name}")
    # subprocess.run([
    #     "python", "scripts/visualize.py", object_name, "--compare"
    # ], check=True, env=env)
