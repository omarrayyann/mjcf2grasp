import json
import os
import subprocess
import wandb
from datetime import datetime

MAX_SUCCESSFUL_GRASPS = 5000
USE_WANDB = 0
objects_list_path = "results/static_objects_list.json"
with open(objects_list_path, "r") as f:
    data = json.load(f)

if USE_WANDB:
    wandb.init(project="thor-grasp-pipeline", name=f"grasp-processing-{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
               config={"total_objects": len(data), "max_successful_grasps": MAX_SUCCESSFUL_GRASPS})
    wandb.define_metric("step")
    wandb.define_metric("completion_percentage", step_metric="step")
    wandb.define_metric("processed_objects", step_metric="step")
    wandb.define_metric("remaining_objects", step_metric="step")
    wandb.define_metric("grasp_count", step_metric="step")
    wandb.define_metric("filtered_count", step_metric="step")
    wandb.define_metric("filter_success_rate", step_metric="step")
    wandb.log({"total_objects": len(data), "step": 0})

processed_objects = 0
failed_objects = []

for obj in data:
    object_name = obj["name"]
    xml_file_path = obj["xml"]
    object_output_dir = os.path.join("results/static_objects", object_name)
    os.makedirs(object_output_dir, exist_ok=True)
    mesh_path = os.path.join(object_output_dir, f"{object_name}_combined.obj")
    manifold_path = os.path.join(object_output_dir, f"{object_name}_manifold.obj")
    simplify_path = os.path.join(object_output_dir, f"{object_name}_simplified.obj")
    grasp_file_path = os.path.join(object_output_dir, f"{object_name}_grasps.json")
    non_filtered_viz_path = os.path.join(object_output_dir, f"{object_name}_grasps_9shot.png")
    filtered_viz_path = os.path.join(object_output_dir, f"{object_name}_filtered_grasps_9shot.png")
    filtered_file_path = os.path.join(object_output_dir, f"{object_name}_grasps_filtered.json")
    if os.path.exists(filtered_file_path):
        print(f"Filtered grasps already exist for {object_name}, skipping to next object")
        processed_objects += 1
        continue
    if not os.path.exists(mesh_path):
        print(f"Converting XML to OBJ for {object_name}")
        try:
            subprocess.run(["python", "pipelines/static/0_generate_mesh.py", xml_file_path, mesh_path], check=True)
            print(f"Created combined mesh: {mesh_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting XML to OBJ for {object_name}: {str(e)}")
            failed_objects.append(object_name)
            continue
    if not os.path.exists(manifold_path):
        print(f"\nProcessing object: {object_name} manifold (using combined mesh from XML)")
        subprocess.run(["./manifold", os.path.abspath(mesh_path), os.path.abspath(manifold_path), "-s"], 
                      cwd="external_src/Manifold/build", check=True)
        print(f"\nProcessing object: {object_name} simplification")
        simplify_success = False
        try:
            subprocess.run(["./simplify", "-i", os.path.abspath(manifold_path), "-o", os.path.abspath(simplify_path), 
                          "-m", "-r", "0.5"], cwd="external_src/Manifold/build", check=True)
            simplify_success = True
        except subprocess.CalledProcessError as e:
            print(f"Simplify failed for {object_name}, retrying manifold and simplify...")
            failed_objects.append(object_name)
            continue
    if not os.path.exists(grasp_file_path):
        print(f"Generating grasps for object: {object_name}")
        try:
            subprocess.run(["python", "pipelines/static/1_generate_grasps.py", "--object_file", simplify_path, 
                          "--quality", "antipodal", "--output", grasp_file_path, "--systematic_sampling", 
                          "--num_workers", str(os.cpu_count()//2)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error generating grasps for {object_name}: {str(e)}")
            failed_objects.append(object_name)
            continue
    if not os.path.exists(non_filtered_viz_path):
        print(f"Visualizing grasps for object: {object_name}")
        visualization_success = False
        try:
            subprocess.run(["python", "scripts/visualize.py", "--grasps_path", grasp_file_path, 
                          "--save-png", non_filtered_viz_path, "--grasp-shape-only"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Visualization for {object_name} failed: {str(e)}")
            print("Continuing to next object...")
            failed_objects.append(object_name)
    if USE_WANDB and os.path.exists(non_filtered_viz_path):
        wandb.log({
            f"{object_name}/images/non_filtered_grasps": wandb.Image(
                non_filtered_viz_path, caption=f"Non-filtered grasps for {object_name}"
            )
        })
    xml_mesh_file_path = xml_file_path.replace(".xml", "_mesh.xml")
    if not os.path.exists(xml_mesh_file_path):
        xml_mesh_file_path = xml_file_path
    if not os.path.exists(filtered_file_path):
        print(f"Filtering grasps for object: {object_name} using MuJoCo")
        try:
            subprocess.run(["python", "pipelines/static/2_filter_mujoco.py", "--object_name", object_name, 
                          "--grasps_path", grasp_file_path, "--xml_file", xml_mesh_file_path, "--num_workers", str(os.cpu_count()//2), 
                          "--approach_distance", "0.3", "--approach_steps", "3000", "--shake_magnitude", "0.1", 
                          "--shake_steps", "1000", 
                          
                          "--max_contact_depth", "1.0",
                          "--min_contact_depth", "0.0",
                          "--center_contact_depth", "0.75",
                          "--contact_depth_bias", "3.0",
                          
                          "--render", "--rotate", "--max_successful", str(MAX_SUCCESSFUL_GRASPS)], 
                          check=True)
        except subprocess.CalledProcessError as e:
            print(f"   Warning: Failed to filter grasps for {object_name}: {str(e)}")
            continue
    if not os.path.exists(filtered_viz_path):
        print(f"Visualizing filtered grasps for object: {object_name}")
        try:
            subprocess.run(["python", "scripts/visualize.py", "--grasps_path", filtered_file_path, 
                          "--save-png", filtered_viz_path, "--grasp-shape-only"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Visualization for {object_name} failed: {str(e)}")
            failed_objects.append(object_name)
    if USE_WANDB and os.path.exists(filtered_viz_path):
        wandb.log({
            f"{object_name}/images/filtered_grasps": wandb.Image(
                filtered_viz_path, caption=f"Filtered grasps for {object_name}"
            )
        })
    grasp_count = 0
    filtered_count = 0
    if os.path.exists(filtered_file_path):
        try:
            with open(filtered_file_path, "r") as f:
                filtered_data = json.load(f)
                filtered_count = len(filtered_data.get("transforms", []))
        except:
            pass
    processed_objects += 1
    filter_success_rate = (filtered_count / grasp_count * 100) if grasp_count > 0 else 0
    completion_percentage = (processed_objects / len(data)) * 100

    progress_bar_width = 50
    filled_width = int(progress_bar_width * (processed_objects / len(data)))
    progress_bar = "█" * filled_width + "░" * (progress_bar_width - filled_width)
    print(f"Progress: [{progress_bar}] {completion_percentage:.1f}% ({processed_objects}/{len(data)})")
    print(f"Completed processing for {object_name}. All files saved in: {object_output_dir}")
    if grasp_count > 0:
        print(f"  - Original grasps: {grasp_count}")
    if filtered_count > 0:
        print(f"  - Filtered grasps: {filtered_count}")
    print("=" * 80)

print(f"\n{'=' * 80}")
print(f"PIPELINE COMPLETE!")
print(f"{'=' * 80}")
print(f"Total objects processed: {processed_objects}/{len(data)}")
if failed_objects:
    print(f"Failed visualizations: {len(failed_objects)}")
    print(f"Failed objects: {', '.join(failed_objects)}")

if USE_WANDB:
    wandb.log({"pipeline_complete": True, "total_processed": processed_objects, 
              "total_failed_visualizations": len(failed_objects), 
              "success_rate": (processed_objects - len(failed_objects)) / processed_objects * 100 if processed_objects > 0 else 0})