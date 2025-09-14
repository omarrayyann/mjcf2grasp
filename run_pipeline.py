import json
import os
import subprocess
import wandb
from datetime import datetime

USE_WANDB = False
gripper_name = "rum"

with open("matched_objs.json", "r") as f:
    data = json.load(f)
if USE_WANDB:
    wandb.init(
        project="thor-grasp-pipeline",
        name=f"grasp-processing-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "total_objects": len(data),
            "approach_distance": 0.1,
            "approach_steps": 1000,
            "max_successful_grasps": 1000,
            "max_grasps_to_visualize": 2000,
        },
    )

    wandb.define_metric("step")
    wandb.define_metric("completion_percentage", step_metric="step")
    wandb.define_metric("processed_objects", step_metric="step")
    wandb.define_metric("remaining_objects", step_metric="step")
    wandb.define_metric("grasp_count", step_metric="step")
    wandb.define_metric("filtered_count", step_metric="step")
    wandb.define_metric("filter_success_rate", step_metric="step")

    print(f"Starting processing of {len(data)} objects")
    print(f"Monitor progress at: {wandb.run.url}")
    wandb.log({"total_objects": len(data), "step": 0})
else:
    print(f"Starting processing of {len(data)} objects (wandb disabled)")

thor_assets_path = "/scratch/olr7742/ai2/thor-grasp/assets/Thor-Assets"
base_input_path = "../../../assets/objects"
temp_folder = "tmp"
os.makedirs(temp_folder, exist_ok=True)
temp_path = os.path.join(temp_folder, "temp.obj")
temp_abs_path = os.path.abspath(temp_path)
output_path = os.path.join(temp_folder, "output.obj")
output_abs_path = os.path.abspath(output_path)

processed_objects = 0
failed_objects = []
total_steps_skipped = 0

for obj in data:
    object_name = obj["name"]
    xml_file_path = obj["xml"]

    print(f"Converting XML to OBJ for {object_name}")
    try:
        subprocess.run(
            ["python", "pipeline/0_generate_mesh.py", xml_file_path, temp_path],
            check=True,
        )
        print(f"✓ Successfully created combined mesh: {temp_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting XML to OBJ for {object_name}: {str(e)}")
        failed_objects.append(object_name)
        continue

    object_output_dir = os.path.join("output", object_name)
    os.makedirs(object_output_dir, exist_ok=True)
    print(f"Created output directory: {object_output_dir}")

    grasp_file_path = os.path.join(object_output_dir, f"{object_name}_grasps.json")
    filtered_file_path = os.path.join(
        object_output_dir, f"{object_name}_grasps_filtered.json"
    )
    filtered_viz_path = os.path.join(
        object_output_dir, f"{object_name}_filtered_grasps_9shot.png"
    )

    old_grasp_file = f"output/{object_name}_grasps.json"
    old_filtered_file = f"output/{object_name}_grasps_filtered.json"

    steps_to_skip = []
    if os.path.exists(grasp_file_path) or os.path.exists(old_grasp_file):
        steps_to_skip.append("grasp generation")
    if os.path.exists(filtered_file_path) or os.path.exists(old_filtered_file):
        steps_to_skip.append("filtering")
    if os.path.exists(filtered_viz_path):
        steps_to_skip.append("visualization")

    if steps_to_skip:
        print(f"⚡ Will skip: {', '.join(steps_to_skip)} (files already exist)")
        total_steps_skipped += len(steps_to_skip)
    else:
        print(
            f"🔄 Will run all steps: manifold processing, grasp generation, filtering, and visualization"
        )

    if USE_WANDB:
        wandb.log(
            {
                "current_object": object_name,
                "progress": (processed_objects + 1) / len(data),
                "processed_count": processed_objects + 1,
                "steps_to_skip": len(steps_to_skip),
                "total_steps": 4,
            }
        )

    print(f"\nProcessing object: {object_name} manifold (using combined mesh from XML)")
    subprocess.run(
        ["./manifold", temp_abs_path, temp_abs_path, "-s"],
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
                temp_abs_path,
                "-o",
                output_abs_path,
                "-m",
                "-r",
                "0.5",
            ],
            cwd="external_src/Manifold/build",
            check=True,
        )
        simplify_success = True
    except subprocess.CalledProcessError as e:
        print(f"✗ Simplify failed for {object_name}, retrying manifold and simplify...")

        try:
            subprocess.run(
                ["./manifold", temp_abs_path, temp_abs_path, "-s"],
                cwd="external_src/Manifold/build",
                check=True,
            )

            subprocess.run(
                [
                    "./simplify",
                    "-i",
                    temp_abs_path,
                    "-o",
                    output_abs_path,
                    "-m",
                    "-r",
                    "0.5",
                ],
                cwd="external_src/Manifold/build",
                check=True,
            )
            simplify_success = True
        except subprocess.CalledProcessError as e2:
            print(
                f"✗ Simplify still failed for {object_name} after re-running manifold: {str(e2)}"
            )
            failed_objects.append(object_name)

    if not simplify_success:
        continue

    if os.path.exists(grasp_file_path):
        print(
            f"✓ Grasp file already exists for {object_name}, skipping grasp generation"
        )
    else:
        print(f"Generating grasps for object: {object_name}")
        try:
            subprocess.run(
                [
                    "python",
                    "pipeline/1_generate_grasps.py",
                    "--object_file",
                    output_abs_path,
                    "--quality",
                    "antipodal",
                    "--output",
                    grasp_file_path,
                    "--systematic_sampling",
                    "--num_workers",
                    str(os.cpu_count()),
                    "--gripper",
                    gripper_name,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error generating grasps for {object_name}: {str(e)}")
            print("Trying again with fewer workers...")
            num_workers = max(1, os.cpu_count() // 2)
            subprocess.run(
                [
                    "python",
                    "pipeline/1_generate_grasps.py",
                    "--object_file",
                    output_abs_path,
                    "--quality",
                    "antipodal",
                    "--output",
                    grasp_file_path,
                    "--systematic_sampling",
                    "--num_workers",
                    str(num_workers),
                    "--gripper",
                    gripper_name,
                ],
                check=True,
            )

    # print(f"Visualizing initial grasps for object: {object_name}")
    # subprocess.run(
    #     [
    #         "python",
    #         "scripts/visualize_render.py",
    #         object_name,
    #         "--render",
    #         "--grasp-shape-only",
    #     ],
    #     check=True,
    # )

    ignore_types = [
        "pen",
        "pencil",
        # too thin
        "plate",
        "key",
        "cd",
        "book",
        "phone",
        "card",
        "bedsheet",
        "lamp",  # not likely to pickup...
        # "pillow",
        "spoon",
        "fork",
        # "plant",  # not likely to pickup...
        # boxy objects that are better prim description
        "laptop",
        "box",
        "statue",  # some have very curvy bottom that it cannot stand
        # furntiure with receptacles with objects on top or inside
        "bed",
        "shelving",
        "table",
        "dresser",
        "desk",
    ]

    # Skip this object if its name contains any of the ignore types
    should_skip = False
    for ignore_type in ignore_types:
        if ignore_type in object_name.lower():
            should_skip = True
            print(
                f"Skipping {object_name} because it contains ignored type: {ignore_type}"
            )
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
            except subprocess.CalledProcessError as e:
                print(
                    f"   Warning: Failed to convert to mesh colliders, using original XML: {str(e)}"
                )
                xml_mesh_file = xml_file_path
            except Exception as e:
                print(
                    f"   Warning: Unexpected error in mesh collider conversion, using original XML: {str(e)}"
                )
                xml_mesh_file = xml_file_path
        else:
            print(f"   Mesh collider XML already exists: {xml_mesh_file}")
    else:
        xml_mesh_file = xml_file_path

    if os.path.exists(filtered_file_path):
        print(
            f"✓ Filtered grasps file already exists for {object_name}, skipping filtering"
        )
    else:
        print(f"Filtering grasps for object: {object_name} using MuJoCo")
        try:
            subprocess.run(
                [
                    "python",
                    "pipeline/3_filter_mujoco.py",
                    "--object_name",
                    object_name,
                    "--grasps_path",
                    grasp_file_path,
                    "--xml_file",
                    xml_mesh_file,
                    "--num_workers",
                    str(os.cpu_count()),
                    "--approach_distance",
                    "0.3",
                    "--approach_steps",
                    "1000",
                    "--shake_magnitude",
                    "0.15",
                    # "--render",
                    "--max_successful",
                    "5000",
                    "--gripper",
                    gripper_name,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error filtering grasps for {object_name}: {str(e)}")
            print("Trying again with fewer workers...")
            num_workers = max(1, os.cpu_count() // 2)
            subprocess.run(
                [
                    "python",
                    "pipeline/3_filter_mujoco.py",
                    "--object_name",
                    object_name,
                    "--grasps_path",
                    grasp_file_path,
                    "--xml_file",
                    xml_file_path,
                    "--num_workers",
                    str(num_workers),
                    "--approach_distance",
                    "0.1",
                    "--approach_steps",
                    "1000",
                    "--shake_magnitude",
                    "0.15",
                    "--render",
                    "--max_successful",
                    "5000",
                    "--gripper",
                    gripper_name,
                ],
                check=True,
            )

    if os.path.exists(filtered_viz_path):
        print(
            f"✓ Visualization already exists for {object_name}, skipping visualization"
        )
        visualization_success = True
    else:
        print(f"Visualizing filtered grasps for object: {object_name}")
        visualization_success = False
        try:
            subprocess.run(
                [
                    "python",
                    "scripts/visualize.py",
                    object_name,
                    "--filtered",
                    "--save-png",
                    filtered_viz_path,
                    # "--no-render",
                    "--grasp-shape-only",
                ],
                check=True,
            )

            visualization_success = True

        except subprocess.CalledProcessError as e:
            print(f"Warning: Visualization for {object_name} failed: {str(e)}")
            print("Continuing to next object...")
            failed_objects.append(object_name)

    if USE_WANDB and os.path.exists(filtered_viz_path):
        wandb.log(
            {
                f"{object_name}_filtered_grasps": wandb.Image(
                    filtered_viz_path, caption=f"Filtered grasps for {object_name}"
                ),
                f"visualization_success": visualization_success,
            }
        )

    if not visualization_success:
        if USE_WANDB:
            wandb.log(
                {
                    "failed_visualization": object_name,
                    "error": str(e) if "e" in locals() else "Unknown error",
                    "visualization_success": False,
                }
            )

    old_grasp_file = f"output/{object_name}_grasps.json"
    old_filtered_file = f"output/{object_name}_grasps_filtered.json"

    grasp_count = 0
    filtered_count = 0

    if os.path.exists(old_grasp_file) and not os.path.exists(grasp_file_path):
        os.rename(old_grasp_file, grasp_file_path)
        print(f"Moved {old_grasp_file} to {grasp_file_path}")

    if os.path.exists(old_filtered_file) and not os.path.exists(filtered_file_path):
        os.rename(old_filtered_file, filtered_file_path)
        print(f"Moved {old_filtered_file} to {filtered_file_path}")

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

    print(
        f"Progress: [{progress_bar}] {completion_percentage:.1f}% ({processed_objects}/{len(data)})"
    )
    print(
        f"Completed processing for {object_name}. All files saved in: {object_output_dir}"
    )
    if grasp_count > 0:
        print(f"  - Original grasps: {grasp_count}")
    if filtered_count > 0:
        print(f"  - Filtered grasps: {filtered_count}")
    print("=" * 80)

print(f"\n{'=' * 80}")
print(f"PIPELINE COMPLETE!")
print(f"{'=' * 80}")
print(f"Total objects processed: {processed_objects}/{len(data)}")
print(f"Total steps skipped: {total_steps_skipped} (due to resumable pipeline)")
if failed_objects:
    print(f"Failed visualizations: {len(failed_objects)}")
    print(f"Failed objects: {', '.join(failed_objects)}")

estimated_time_per_step = 2
estimated_time_saved = total_steps_skipped * estimated_time_per_step
if estimated_time_saved > 0:
    print(
        f"⏱️  Estimated time saved: ~{estimated_time_saved} minutes ({estimated_time_saved / 60:.1f} hours)"
    )

if USE_WANDB:
    wandb.log(
        {
            "pipeline_complete": True,
            "total_processed": processed_objects,
            "total_failed_visualizations": len(failed_objects),
            "total_steps_skipped": total_steps_skipped,
            "estimated_time_saved_minutes": total_steps_skipped * 2,
            "success_rate": (processed_objects - len(failed_objects))
            / processed_objects
            * 100
            if processed_objects > 0
            else 0,
        }
    )

summary_data = []


for obj in data[:processed_objects]:
    object_name = os.path.splitext(os.path.basename(obj["path"]))[0]
    object_dir = os.path.join("output", object_name)

    grasp_file = os.path.join(object_dir, f"{object_name}_grasps.json")
    filtered_file = os.path.join(object_dir, f"{object_name}_grasps_filtered.json")

    grasp_count = 0
    filtered_count = 0

    if os.path.exists(grasp_file):
        try:
            with open(grasp_file, "r") as f:
                data_content = json.load(f)
                grasp_count = len(data_content.get("transforms", []))
        except:
            pass

    if os.path.exists(filtered_file):
        try:
            with open(filtered_file, "r") as f:
                data_content = json.load(f)
                filtered_count = len(data_content.get("transforms", []))
        except:
            pass

    summary_data.append(
        [
            object_name,
            grasp_count,
            filtered_count,
            f"{filtered_count / grasp_count * 100:.1f}%" if grasp_count > 0 else "0%",
            "✓" if object_name not in failed_objects else "✗",
        ]
    )

if USE_WANDB:
    table = wandb.Table(
        columns=[
            "Object",
            "Original Grasps",
            "Filtered Grasps",
            "Success Rate",
            "Visualization",
        ],
        data=summary_data,
    )
    wandb.log({"processing_summary": table})

    wandb.finish()
    print("Results logged to Weights & Biases!")
else:
    print("Pipeline completed (wandb was disabled)")
