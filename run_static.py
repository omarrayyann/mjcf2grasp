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
    wandb.init(
        project="thor-grasp-pipeline",
        name=f"grasp-processing-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "total_objects": len(data),
            "max_successful_grasps": MAX_SUCCESSFUL_GRASPS,
        },
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
            subprocess.run(
                ["python", "pipelines/static/0_generate_mesh.py", xml_file_path, mesh_path],
                check=True,
            )
            print(f"Created combined mesh: {mesh_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting XML to OBJ for {object_name}: {str(e)}")
            failed_objects.append(object_name)
            continue
    
    if not os.path.exists(manifold_path):
        print(f"\nProcessing object: {object_name} manifold (using combined mesh from XML)")
        subprocess.run(
            ["./manifold", mesh_path, manifold_path, "-s"],
            cwd="external_src/Manifold/build",
            check=True,
        )

        print(f"\nProcessing object: {object_name} simplification")
        simplify_success = False
        try:
            subprocess.run(
                [
                    "./simplify",
                    "-i",
                    manifold_path,
                    "-o",
                    simplify_path,
                    "-m",
                    "-r",
                    "0.5",
                ],
                cwd="external_src/Manifold/build",
                check=True,
            )
            simplify_success = True
        except subprocess.CalledProcessError as e:
            print(f"Simplify failed for {object_name}, retrying manifold and simplify...")
            failed_objects.append(object_name)
            continue

    if not simplify_success:
        continue

    if os.path.exists():
        print("Grasp file already exists for {object_name}, skipping grasp generation")
    else:
        print(f"Generating grasps for object: {object_name}")
        try:
            subprocess.run(
                [
                    "python",
                    "pipelines/static/1_generate_grasps.py",
                    "--object_file",
                    simplify_path,
                    "--quality",
                    "antipodal",
                    "--output",
                    grasp_file_path,
                    "--systematic_sampling",
                    "--num_workers",
                    1,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error generating grasps for {object_name}: {str(e)}")
            failed_objects.append(object_name)
            continue

    if not os.path.exists(non_filtered_viz_path):
        print(f"Visualizing grasps for object: {object_name}")
        visualization_success = False
        try:
            subprocess.run(
                [
                    "python",
                    "scripts/visualize.py",
                    "--grasps_path",
                    grasp_file_path,
                    "--save-png",
                    filtered_viz_path,
                    "--grasp-shape-only",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Visualization for {object_name} failed: {str(e)}")
            print("Continuing to next object...")
            failed_objects.append(object_name)

    ignore_types = [
        "pen",
        "pencil",
        "plate",
        "key",
        "cd",
        "book",
        "phone",
        "card",
        "bedsheet",
        "lamp", 
        "spoon",
        "fork",
        "laptop",
        "box",
        "statue",
        "bed",
        "shelving",
        "table",
        "dresser",
        "desk",
    ]

    should_skip = False
    for ignore_type in ignore_types:
        if ignore_type in object_name.lower():
            should_skip = True
            break

    if not should_skip:
        xml_mesh_file = xml_file_path.replace(".xml", "_mesh.xml")
        if not os.path.exists(xml_mesh_file):
            print(f"   Converting XML to use mesh colliders...")
            try:
                subprocess.run(
                    [
                        "python",
                        "pipeline_articulate/2_mesh_colliders.py",
                        "--input",
                        xml_file_path,
                        "--output",
                        xml_mesh_file,
                    ],
                    check=True,
                )
                print(f"   Mesh collider XML created: {xml_mesh_file}")
            except Exception as e:
                print(f"   Warning: Unexpected error in mesh collider conversion, using original XML: {str(e)}")
                xml_mesh_file = xml_file_path
    else:
        xml_mesh_file = xml_file_path

    if not os.path.exists(filtered_file_path):
        print(f"Filtering grasps for object: {object_name} using MuJoCo")
        try:
            subprocess.run(
                [
                    "python",
                    "pipelines/static/3_filter_mujoco.py",
                    "--object_name",
                    object_name,
                    "--grasps_path",
                    grasp_file_path,
                    "--xml_file",
                    xml_mesh_file,
                    "--num_workers",
                    str(1),
                    "--approach_distance",
                    "0.3",
                    "--approach_steps",
                    "3000",
                    "--shake_magnitude",
                    "0.1",
                    "--shake_steps",
                    "1000",
                    "--rotate",
                    # "--render",
                    "--max_successful",
                    "5000",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"   Warning: Failed to filter grasps for {object_name}: {str(e)}")
            continue

    if not os.path.exists(filtered_viz_path):
        print(f"Visualizing filtered grasps for object: {object_name}")
        try:
            subprocess.run(
                [
                    "python",
                    "scripts/visualize.py",
                    "--grasps_path",
                    filtered_file_path,
                    "--save-png",
                    filtered_viz_path,
                    "--grasp-shape-only",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Visualization for {object_name} failed: {str(e)}")
            failed_objects.append(object_name)

    if USE_WANDB and os.path.exists(filtered_viz_path):
        wandb.log(
            {f"filtered_visualization": wandb.Image(filtered_viz_path), "object_name": object_name}
        )

    if not visualization_success:
        if USE_WANDB:
            wandb.log(
                {f"visualization_failure_{object_name}": True, "object_name": object_name}
            )

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

    if USE_WANDB:
        wandb.log(
            {
                "completion_percentage": completion_percentage,
                "processed_objects": processed_objects,
                "remaining_objects": len(data) - processed_objects,
                "overall_progress": processed_objects / len(data),
                f"object_completed": object_name,
                f"grasp_count": grasp_count,
                f"filtered_count": filtered_count,
                f"filter_success_rate": filter_success_rate,
                f"visualization_success": visualization_success,
                "step": processed_objects,
            }
        )

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
    wandb.log(
        {
            "pipeline_complete": True,
            "total_processed": processed_objects,
            "total_failed_visualizations": len(failed_objects),
            "success_rate": (processed_objects - len(failed_objects))
            / processed_objects
            * 100
            if processed_objects > 0
            else 0,
        }
    )