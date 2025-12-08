import json
import os
import subprocess
import wandb
import argparse
import numpy as np

MAX_SUCCESSFUL_GRASPS = 1000
USE_WANDB = 1
WANDB_RUN_NAME = "shared-grasp-processing-final"
NUM_WORKERS = 5 # os.cpu_count()

parser = argparse.ArgumentParser(description='Process static objects for grasp generation')
parser.add_argument('--objects_list', type=str, default="results/static_objects_list.json", help='Path to the JSON file containing the list of objects')
args = parser.parse_args()

if not os.path.exists(args.objects_list):
    raise FileNotFoundError(f"Objects list file not found: {args.objects_list}")

with open(args.objects_list, "r") as f:
    data = json.load(f)

print(f"Total objects in dataset: {len(data)}")

if USE_WANDB:
    run_id = args.wandb_run_id if args.wandb_run_id else WANDB_RUN_NAME
    
    wandb.init(
        project="thor-grasp-pipeline", 
        name=WANDB_RUN_NAME,
        id=run_id,
        resume="allow",
        config={"total_objects": len(data), "max_successful_grasps": MAX_SUCCESSFUL_GRASPS}
    )
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
    filtered_viz_path = os.path.join(object_output_dir, f"{object_name}_filtered_grasps_9shot.png")
    filtered_npz_path = os.path.join(object_output_dir, f"{object_name}_grasps_filtered.npz")
    filtered_json_path = os.path.join(object_output_dir, f"{object_name}_grasps_object_info.json")
    
    if os.path.exists(filtered_npz_path):
        processed_objects += 1
        print(f"Object {object_name} already fully processed, skipping...")
        continue
    
    print(f"Starting processing for {object_name}")
    
    processing_failed = False
    failure_reason = ""
    
    try:
        if not os.path.exists(mesh_path):
            print(f"Converting XML to OBJ for {object_name}")
            try:
                subprocess.run(["python", "pipeline/combine_meshes.py", xml_file_path, mesh_path, "--only_collision"], check=True)
                print(f"Created combined mesh: {mesh_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting XML to OBJ for {object_name}: {str(e)}")
                processing_failed = True
                failure_reason = "mesh_generation_failed"
                failed_objects.append(object_name)
                continue
                
        if not os.path.exists(manifold_path):
            print(f"\nProcessing object: {object_name} manifold (using combined mesh from XML)")
            try:
                subprocess.run(["./manifold", os.path.abspath(mesh_path), os.path.abspath(manifold_path), "-s"], 
                              cwd="external_src/Manifold/build", check=True)
            except subprocess.CalledProcessError as e:
                print(f"Manifold failed for {object_name}: {str(e)}")
                processing_failed = True
                failure_reason = "manifold_failed"
                failed_objects.append(object_name)
                continue
            
            print(f"\nProcessing object: {object_name} simplification")
            try:
                subprocess.run(["./simplify", "-i", os.path.abspath(manifold_path), "-o", os.path.abspath(simplify_path), 
                              "-m", "-r", "0.5"], cwd="external_src/Manifold/build", check=True)
            except subprocess.CalledProcessError as e:
                print(f"Simplify failed for {object_name}: {str(e)}")
                processing_failed = True
                failure_reason = "simplify_failed"
                failed_objects.append(object_name)
                continue
        
        if not os.path.exists(grasp_file_path):
            try:
                subprocess.run(["python", "pipeline/generate_grasps.py", "--object_file", simplify_path, 
                              "--quality", "antipodal", "--output", grasp_file_path, 
                            #   "--systematic_sampling", #TODO: return
                              "--num_workers", str(NUM_WORKERS)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error generating grasps for {object_name}: {str(e)}")
                processing_failed = True
                failure_reason = "grasp_generation_failed"
                failed_objects.append(object_name)
                continue
                
        xml_mesh_file_path = xml_file_path.replace(".xml", "_mesh.xml")
        if not os.path.exists(xml_mesh_file_path):
            xml_mesh_file_path = xml_file_path
            
        if not os.path.exists(filtered_npz_path):
            print(f"Filtering grasps for object: {object_name} using MuJoCo")
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    if attempt > 1:
                        print(f"Retry attempt {attempt}/{max_attempts} (without diversity mode)...")
                    
                    cmd_args = ["python", "pipeline/perturbations_test.py", 
                               "--object_name", object_name, 
                               "--grasps_path", grasp_file_path, 
                               "--xml_file", xml_mesh_file_path, 
                               "--approach_distance", "0.3", 
                               "--approach_steps", "3000", 
                               "--shake_magnitude", "0.1", 
                               "--shake_steps", "1000", 
                               "--max_contact_depth", "1.0",
                               "--min_contact_depth", "0.0",
                               "--center_contact_depth", "0.75",
                               "--contact_depth_bias", "2.8",
                               "--num_workers", str(NUM_WORKERS),
                               "--rotate", 
                               "--max_successful", str(MAX_SUCCESSFUL_GRASPS)]
                    
                    if attempt == 1:
                        cmd_args.append("--diversity_mode")
                    
                    subprocess.run(cmd_args, check=True)
                    break
                except subprocess.CalledProcessError as e:
                    print(f"  Error output: {e.stderr}")
                    if attempt == max_attempts:
                        print(f"   Warning: Failed to filter grasps for {object_name} after {max_attempts} attempts")
                        print(f"   Return code: {e.returncode}")
                        processing_failed = True
                        failure_reason = "filter_failed"
                        failed_objects.append(object_name)
                        continue
                    else:
                        print(f"Attempt {attempt} failed, retrying...")
        else:
            print("File already exist")
            
        grasp_count = 0
        filtered_count = 0
        if os.path.exists(filtered_npz_path):
            try:
                npz_data = np.load(filtered_npz_path)
                filtered_count = len(npz_data["transforms"])
            except:
                pass
            for f in os.listdir(object_output_dir):
                file_path = os.path.join(object_output_dir, f)
                if file_path != filtered_npz_path and os.path.isfile(file_path):
                    os.remove(file_path)
                
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
        
    except Exception as e:
        print(f"Failed with {e}")


print(f"Total objects in dataset: {len(data)}")
print(f"Objects successfully processed by this job: {processed_objects}")
if failed_objects:
    print(f"Failed: {len(failed_objects)}")
    if len(failed_objects) <= 10:
        print(f"Failed objects: {', '.join(failed_objects)}")
    else:
        print(f"Failed objects (first 10): {', '.join(failed_objects[:10])}")

if USE_WANDB:
    wandb.log({"pipeline_complete": True, "total_processed": processed_objects, 
              "total_failed": len(failed_objects), 
              "success_rate": (processed_objects - len(failed_objects)) / processed_objects * 100 if processed_objects > 0 else 0})