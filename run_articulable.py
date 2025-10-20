import json
import os
import subprocess
import sys
import wandb
from datetime import datetime
from pathlib import Path

USE_WANDB = True
gripper = "robotiq"

def load_articulated_objects():
    matched_file = "results/articulable_objects_list.json"

    if not os.path.exists(matched_file):
        print(f"Error: {matched_file} not found!")
        print(
            "Please run 'python scripts/find_objects.py' first to generate object list"
        )
        return []

    with open(matched_file, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} articulatable objects for processing")
    return data


def run_grasp_filtering_stage(
    object_name, grasps_path, xml_file, output_dir, per_joint_grasps_json=None
):
    # First, convert XML to use mesh colliders
    xml_mesh_file = xml_file.replace(".xml", "_mesh.xml")
    if not os.path.exists(xml_mesh_file):
        print(f"   Converting XML to use mesh colliders...")
        try:
            subprocess.run(
                [
                    "python",
                    "pipelines/articulable/2_mesh_colliders.py",
                    "--input",
                    xml_file,
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
            xml_mesh_file = xml_file
        except Exception as e:
            print(
                f"   Warning: Unexpected error in mesh collider conversion, using original XML: {str(e)}"
            )
            xml_mesh_file = xml_file
    else:
        print(f"   Mesh collider XML already exists: {xml_mesh_file}")

    # Use the mesh collider XML for filtering
    xml_file_for_filtering = xml_mesh_file

    if per_joint_grasps_json:
        print(f"   Per-joint grasps JSON: {per_joint_grasps_json}")
        try:
            print(f"   Filtering per-joint grasps using MuJoCo simulation...")
            subprocess.run(
                [
                    "python",
                    "pipelines/articulable/3_filter_mujoco.py",
                    "--object_name",
                    object_name,
                    "--per_joint_summary_json",
                    per_joint_grasps_json,
                    "--xml_file",
                    xml_file_for_filtering,
                    "--num_workers",
                    str(os.cpu_count()),
                    "--approach_distance",
                    "0.3",
                    "--approach_steps",
                    "8",
                    "--max_successful",
                    "2000",
                    "--gripper",
                    gripper,
                    # "--render",
                ],
                check=True,
            )
            summary_path = per_joint_grasps_json
            print(f"   Per-joint filtered summary: {summary_path}")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                for entry in summary:
                    filtered_count = entry.get("num_successful_filtered_grasps", 0)
                    grasps_file = entry.get("grasps_file", "")
                    original_count = 0
                    if grasps_file and os.path.exists(
                        os.path.join(os.path.dirname(summary_path), grasps_file)
                    ):
                        with open(
                            os.path.join(os.path.dirname(summary_path), grasps_file),
                            "r",
                        ) as gf:
                            gdata = json.load(gf)
                        original_count = len(gdata.get("transforms", []))
                    success_rate = (
                        (filtered_count / original_count * 100)
                        if original_count > 0
                        else 0
                    )
                    print(
                        f"      Joint: {entry['joint']} | Success: {filtered_count}/{original_count} ({success_rate:.1f}%)"
                    )
            return True, summary_path
        except subprocess.CalledProcessError as e:
            print(f"   Error in per-joint grasp filtering: {str(e)}")
            return False, None
        except Exception as e:
            print(f"   Unexpected error: {str(e)}")
            return False, None
    else:
        print(f"   Grasps file: {grasps_path}")
        print(f"   XML file: {xml_file_for_filtering}")
        filtered_grasps_path = grasps_path.replace(".json", "_filtered.json")
        try:
            print(f"   Filtering grasps using MuJoCo simulation...")
            subprocess.run(
                [
                    "python",
                    "pipelines/articulable/3_filter_mujoco.py",
                    "--object_name",
                    object_name,
                    "--grasps_path",
                    grasps_path,
                    "--xml_file",
                    xml_file_for_filtering,
                    "--num_workers",
                    "10",
                    "--approach_distance",
                    "0.5",
                    "--approach_steps",
                    "5",
                    "--max_successful",
                    "10",
                    "--gripper",
                    gripper,
                    # "--render",
                ],
                check=True,
            )
            print(f"   Filtered grasps saved: {filtered_grasps_path}")
            with open(grasps_path, "r") as f:
                original_grasps = json.load(f)
            with open(filtered_grasps_path, "r") as f:
                filtered_grasps = json.load(f)
            original_count = len(original_grasps.get("transforms", []))
            filtered_count = len(filtered_grasps.get("transforms", []))
            success_rate = (
                (filtered_count / original_count * 100) if original_count > 0 else 0
            )
            print(f"   Original grasps: {original_count}")
            print(f"   Successful grasps: {filtered_count}")
            print(f"   Success rate: {success_rate:.1f}%")
            return True, filtered_grasps_path
        except subprocess.CalledProcessError as e:
            print(f"   Error in grasp filtering: {str(e)}")
            return False, None
        except Exception as e:
            print(f"   Unexpected error: {str(e)}")
            return False, None


def run_grasp_generation_stage(object_name, handle_mesh_path, full_mesh, output_dir):
    print(f"\nStage 2: Grasp Generation for {object_name}")
    print(f"   Handle mesh: {handle_mesh_path}")

    grasps_output_path = os.path.join(output_dir, f"{object_name}_grasps.json")

    if os.path.exists(grasps_output_path):
        print(f"   Grasps already exist, skipping...")
        return True, grasps_output_path

    try:
        print(f"   Generating grasps for handle mesh...")
        subprocess.run(
            [
                "python",
                "pipelines/articulable/1_generate_grasps.py",
                "--object_file",
                handle_mesh_path,
                "--output",
                grasps_output_path,
                "--quality",
                "antipodal",
                "--min_quality",
                "0.001",
                "--systematic_sampling",
                "--classname",
                "articulated_handle",
                "--dataset",
                "thor_articulated",
                "--collision_object_file",
                full_mesh,
                "--gripper",
                gripper,
            ],
            check=True,
        )

        print(f"   Grasps generated: {grasps_output_path}")

        with open(grasps_output_path, "r") as f:
            grasps_data = json.load(f)

        num_grasps = len(grasps_data.get("transforms", []))
        print(f"   Total grasps generated: {num_grasps}")

        if num_grasps > 0:
            print(f"   Gripper: {grasps_data.get('gripper', 'unknown')}")
            print(f"   Object scale: {grasps_data.get('object_scale', 1.0)}")

            print(f"   Visualizing generated grasps for {object_name}...")
            viz_grasps = 1
            if viz_grasps:
                try:
                    subprocess.run(
                        [
                            "python",
                            "scripts/visualize_render.py",
                            object_name,
                            "--render",
                            "--grasp-shape-only",
                            "--articulated",
                        ],
                        check=True,
                    )
                    print(f"   Grasp visualization completed for {object_name}")
                except subprocess.CalledProcessError as e:
                    print(
                        f"   Warning: Grasp visualization failed for {object_name}: {str(e)}"
                    )
        else:
            print(f"   Warning: No valid grasps generated for {object_name}")

        return True, grasps_output_path

    except subprocess.CalledProcessError as e:
        print(f"   Error in grasp generation: {str(e)}")
        return False, None
    except Exception as e:
        print(f"   Unexpected error: {str(e)}")
        return False, None


def run_per_joint_grasp_generation(
    object_name, joint_meshes_json, output_dir, full_mesh
):
    print(f"\nStage 2: Per-Joint Grasp Generation for {object_name}")
    print(f"   Joint meshes JSON: {joint_meshes_json}")
    if isinstance(joint_meshes_json, str):
        with open(joint_meshes_json, "r") as f:
            joint_meshes_data = json.load(f)
    else:
        joint_meshes_data = joint_meshes_json

    if all(
        joint.get("grasps_file")
        and os.path.exists(
            os.path.join(os.path.dirname(joint_meshes_json), joint["grasps_file"])
        )
        for joint in joint_meshes_data
    ):
        print(f"   All grasp files exist, skipping per-joint grasp generation.")
        return True, joint_meshes_json

    try:
        subprocess.run(
            [
                "python",
                "pipelines/articulable/1_generate_grasps.py",
                "--per_joint_grasps_from_meshes",
                joint_meshes_json,
                "--gripper",
                gripper,
                "--quality",
                "antipodal",
                "--min_quality",
                "0.005",
                "--systematic_sampling",
                "--classname",
                "articulated_handle",
                "--dataset",
                "thor_articulated",
                "--collision_object_file",
                full_mesh,
            ],
            check=True,
        )
        print(f"   Per-joint grasps generated and summary updated: {joint_meshes_json}")
        return True, joint_meshes_json
    except subprocess.CalledProcessError as e:
        print(f"   Error in per-joint grasp generation: {str(e)}")
        return False, None
    except Exception as e:
        print(f"   Unexpected error: {str(e)}")
        return False, None


def run_handle_detection_stage(obj, output_dir):
    object_name = obj["name"]
    xml_file_path = obj["xml"]

    print(f"\nStage 0: Handle Detection for {object_name}")
    print(f"   XML: {xml_file_path}")

    full_mesh_path = os.path.join(output_dir, "main.obj")
    handle_mesh_path = None

    if os.path.exists(full_mesh_path):
        print(f"   Full mesh already exists, skipping...")
        return True, handle_mesh_path, full_mesh_path
    try:
        print(f"   Generating meshes with LLM analysis...")
        subprocess.run(
            [
                "python",
                "pipelines/articulable/0_generate_mesh.py",
                xml_file_path,
                os.path.join(output_dir, "main.obj"),
            ],
            check=True,
        )
        print(f"   Meshes created in: {output_dir}")
        return True, handle_mesh_path, full_mesh_path
    except subprocess.CalledProcessError as e:
        print(f"   Error in handle detection: {str(e)}")
        return False, None, None
    except Exception as e:
        print(f"   Unexpected error: {str(e)}")
        return False, None, None


def main():
    print("ARTICULATED OBJECT GRASPING PIPELINE")
    print("=" * 50)
    print("This pipeline focuses on generating grasps for manipulating")
    print("articulated objects through handle interaction.")
    print("")
    print("Current Stages:")
    print("  Stage 0: LLM-based handle detection and mesh extraction")
    print("  Stage 1: Joint axis analysis for primary articulation")
    print("  Stage 2: Grasp generation for handle meshes")
    print("  Stage 3: Grasp filtering using MuJoCo simulation")
    print("=" * 50)

    articulated_objects = load_articulated_objects()

    if not articulated_objects:
        print("No articulatable objects found to process!")
        return 1

    # Initialize wandb if enabled
    if USE_WANDB:
        wandb.init(
            project="thor-articulated-grasp-pipeline",
            name=f"articulated-processing-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "total_objects": len(articulated_objects),
                "approach_distance": 0.5,
                "approach_steps": 5,
                "max_successful_grasps": 100,
                "num_workers": 10,
                "pipeline_stages": "0-3",
            },
        )

        wandb.define_metric("step")
        wandb.define_metric("completion_percentage", step_metric="step")
        wandb.define_metric("processed_objects", step_metric="step")
        wandb.define_metric("remaining_objects", step_metric="step")
        wandb.define_metric("grasp_count", step_metric="step")
        wandb.define_metric("filtered_count", step_metric="step")
        wandb.define_metric("filter_success_rate", step_metric="step")

        print(f"Starting processing of {len(articulated_objects)} articulated objects")
        print(f"Monitor progress at: {wandb.run.url}")
        wandb.log({"total_objects": len(articulated_objects), "step": 0})
    else:
        print(
            f"Starting processing of {len(articulated_objects)} articulated objects (wandb disabled)"
        )

    output_base = "results/articulable_objects"
    os.makedirs(output_base, exist_ok=True)

    processed_objects = 0
    failed_objects = []
    successful_objects = []

    for i, obj in enumerate(articulated_objects):
        object_name = obj["name"]

        print(f"\nProcessing {i + 1}/{len(articulated_objects)}: {object_name}")
        print("=" * 60)

        object_output_dir = os.path.join(output_base, object_name)
        os.makedirs(object_output_dir, exist_ok=True)

        # Log current object progress to wandb
        if USE_WANDB:
            wandb.log(
                {
                    "current_object": object_name,
                    "progress": (i + 1) / len(articulated_objects),
                    "processed_count": i + 1,
                    "step": i + 1,
                }
            )

        success, handle_mesh, full_mesh = run_handle_detection_stage(
            obj, object_output_dir
        )

        if not success:
            failed_objects.append(object_name)
            print(f"   Failed at Stage 0 (handle detection) for {object_name}")
            continue

        joint_meshes_json = os.path.join(object_output_dir, f"joint_meshes_info.json")
        per_joint_grasps_success = False
        if os.path.exists(joint_meshes_json):
            per_joint_grasps_success, joint_meshes_json_out = (
                run_per_joint_grasp_generation(
                    object_name, joint_meshes_json, object_output_dir, full_mesh
                )
            )
            if per_joint_grasps_success:
                print(f"   Visualizing each per-joint grasp file for {object_name}...")
                try:
                    with open(joint_meshes_json, "r") as f:
                        joint_grasps_summary = json.load(f)
                    for entry in joint_grasps_summary:
                        grasps_file = os.path.join(
                            os.path.dirname(joint_meshes_json), entry["grasps_file"]
                        )
                        handle_mesh = os.path.join(
                            os.path.dirname(joint_meshes_json), entry["handle_mesh"]
                        )
                        print(
                            f"      Visualizing joint: {entry['joint']} ({grasps_file})"
                        )
                        viz_grasps = False
                        if viz_grasps:
                            try:
                                subprocess.run(
                                    [
                                        "python",
                                        "scripts/visualize_render.py",
                                        entry["joint"],
                                        "--json_file",
                                        grasps_file,
                                        "--render",
                                        "--grasp-shape-only",
                                    ],
                                    check=True,
                                )
                            except subprocess.CalledProcessError as e:
                                print(
                                    f"      Warning: Per-joint grasp visualization failed for {entry['joint']}: {str(e)}"
                                )
                except Exception as e:
                    print(
                        f"   Warning: Could not visualize per-joint grasps individually: {str(e)}"
                    )
                print(
                    f"   Visualizing all per-joint grasps on full mesh for {object_name}..."
                )
                viz_grasps = 0
                if viz_grasps:
                    try:
                        subprocess.run(
                            [
                                "python",
                                "scripts/visualize_all_grasps_on_full_mesh.py",
                                "--grasps_json",
                                joint_meshes_json,
                                "--xml",
                                obj["xml"],
                                "--full_mesh",
                                full_mesh,
                            ],
                            check=True,
                        )
                        print(
                            f"   Per-joint grasp visualization completed for {object_name}"
                        )
                    except subprocess.CalledProcessError as e:
                        print(
                            f"   Warning: Per-joint grasp visualization failed for {object_name}: {str(e)}"
                        )
        else:
            print(f"   Error: joint_meshes.json not found: {joint_meshes_json}")
            print(f"   Cannot proceed with per-joint grasp generation")

        joint_axis_success = False
        joint_axis_path = os.path.join(
            object_output_dir, f"{object_name}_joint_axis.json"
        )

        grasps_success = False
        grasps_path = None

        # if handle_mesh and os.path.exists(handle_mesh):
        #     grasps_success, grasps_path = run_grasp_generation_stage(
        #         object_name, handle_mesh, full_mesh, object_output_dir
        #     )
        # else:
        #     print(f"   Error: Handle mesh not found: {handle_mesh}")
        #     print(f"   Cannot proceed with grasp generation")

        filtering_success = True
        # filtered_grasps_path = f""
        print(f"\nStage 3: Grasp Filtering for {object_name}")
        if per_joint_grasps_success and os.path.exists(joint_meshes_json):
            filtered_grasps_file = "joint_meshes_info_filtered.json"
            if not os.path.exists(
                os.path.join(object_output_dir, filtered_grasps_file)
            ):
                filtering_success, filtered_grasps_path = run_grasp_filtering_stage(
                    object_name,
                    None,
                    obj["xml"],
                    object_output_dir,
                    per_joint_grasps_json=joint_meshes_json,
                )
            else:
                filtering_success = True
                filtered_grasps_path = os.path.join(
                    object_output_dir, filtered_grasps_file
                )
            if (
                filtering_success
                and filtered_grasps_path
                and os.path.exists(filtered_grasps_path)
            ):
                print(
                    f"   Visualizing each filtered per-joint grasp file for {object_name}..."
                )
                viz_grasps = False
                if viz_grasps:
                    try:
                        with open(filtered_grasps_path, "r") as f:
                            filtered_joint_grasps_summary = json.load(f)
                        for entry in filtered_joint_grasps_summary:
                            filtered_grasps_file = os.path.join(
                                os.path.dirname(filtered_grasps_path),
                                entry.get("filtered_grasps_file", ""),
                            )

                            if (
                                not filtered_grasps_file
                                or not filtered_grasps_file.endswith(".json")
                                or not os.path.isfile(filtered_grasps_file)
                            ):
                                continue
                            print(
                                f"      Visualizing joint (filtered): {entry['joint']} ({filtered_grasps_file})"
                            )
                            try:
                                subprocess.run(
                                    [
                                        "python",
                                        "scripts/visualize_render.py",
                                        entry["joint"],
                                        "--json_file",
                                        filtered_grasps_file,
                                        "--render",
                                        "--grasp-shape-only",
                                    ],
                                    check=True,
                                )
                            except subprocess.CalledProcessError as e:
                                print(
                                    f"      Warning: Filtered per-joint grasp visualization failed for {entry['joint']}: {str(e)}"
                                )
                    except Exception as e:
                        print(
                            f"   Warning: Could not visualize filtered per-joint grasps individually: {str(e)}"
                        )
                print(
                    f"   Visualizing all filtered per-joint grasps on full mesh for {object_name}..."
                )
                try:
                    filtered_grasps_path = os.path.join(
                        object_output_dir, "joint_meshes_info_filtered.json"
                    )
                    visualization_png = os.path.join(
                        object_output_dir,
                        f"{object_name}_filtered_grasps_visualization.png",
                    )
                    subprocess.run(
                        [
                            "python",
                            "scripts/visualize_all_grasps_on_full_mesh.py",
                            "--grasps_json",
                            filtered_grasps_path,
                            "--xml",
                            obj["xml"],
                            "--full_mesh",
                            full_mesh,
                            "--filtered_grasps",
                            "--save-png",
                            visualization_png,
                            "--no-render",
                        ],
                        check=True,
                    )
                    print(
                        f"   Filtered per-joint grasp visualization completed for {object_name}"
                    )

                    # Log visualization to wandb if enabled
                    if USE_WANDB and os.path.exists(visualization_png):
                        wandb.log(
                            {
                                f"{object_name}_filtered_grasps_visualization": wandb.Image(
                                    visualization_png,
                                    caption=f"Filtered articulated grasps for {object_name}",
                                ),
                                "visualization_success": True,
                            }
                        )
                except subprocess.CalledProcessError as e:
                    print(
                        f"   Warning: Filtered per-joint grasp visualization failed for {object_name}: {str(e)}"
                    )
                    # Log visualization failure to wandb if enabled
                    if USE_WANDB:
                        wandb.log(
                            {
                                "failed_visualization": object_name,
                                "error": str(e),
                                "visualization_success": False,
                            }
                        )
        elif grasps_success and grasps_path and os.path.exists(grasps_path):
            filtering_success, filtered_grasps_path = run_grasp_filtering_stage(
                object_name, grasps_path, obj["xml"], object_output_dir
            )
        else:
            print(f"   No valid grasps found for filtering for {object_name}")

        if success:
            processed_objects += 1
            successful_objects.append(
                {
                    "name": object_name,
                    "handle_mesh": handle_mesh,
                    "full_mesh": full_mesh,
                    "grasps": grasps_path if grasps_success else None,
                    "filtered_grasps": filtered_grasps_path
                    if filtering_success
                    else None,
                    "joint_axis": joint_axis_path if joint_axis_success else None,
                    "xml": obj["xml"],
                    "output_dir": object_output_dir,
                    "stages_completed": {
                        "handle_detection": True,
                        "joint_axis_analysis": joint_axis_success,
                        "grasp_generation": grasps_success,
                        "grasp_filtering": filtering_success,
                    },
                }
            )

            if filtering_success and joint_axis_success:
                status = "Fully processed (all stages)"
            elif grasps_success and joint_axis_success:
                status = "Partially processed (no filtering)"
            elif joint_axis_success:
                status = "Partially processed (no grasps)"
            else:
                status = "Partially processed (no joint analysis)"
            print(f"   {status}: {object_name}")

            # Log object completion to wandb
            if USE_WANDB:
                # Calculate grasp metrics
                grasp_count = 0
                filtered_count = 0

                if (
                    filtering_success
                    and filtered_grasps_path
                    and os.path.exists(filtered_grasps_path)
                ):
                    try:
                        with open(filtered_grasps_path, "r") as f:
                            filtered_data = json.load(f)
                        for entry in filtered_data:
                            if "filtered_grasps_file" in entry:
                                filtered_file = os.path.join(
                                    object_output_dir, entry["filtered_grasps_file"]
                                )
                                if os.path.exists(filtered_file):
                                    with open(filtered_file, "r") as ff:
                                        filtered_grasp_data = json.load(ff)
                                    filtered_count += len(
                                        filtered_grasp_data.get("transforms", [])
                                    )

                            if "grasps_file" in entry:
                                grasp_file = os.path.join(
                                    object_output_dir, entry["grasps_file"]
                                )
                                if os.path.exists(grasp_file):
                                    with open(grasp_file, "r") as gf:
                                        grasp_data = json.load(gf)
                                    grasp_count += len(grasp_data.get("transforms", []))
                    except:
                        pass

                filter_success_rate = (
                    (filtered_count / grasp_count * 100) if grasp_count > 0 else 0
                )
                completion_percentage = (
                    processed_objects / len(articulated_objects)
                ) * 100

                wandb.log(
                    {
                        "completion_percentage": completion_percentage,
                        "processed_objects": processed_objects,
                        "remaining_objects": len(articulated_objects)
                        - processed_objects,
                        "overall_progress": processed_objects
                        / len(articulated_objects),
                        "object_completed": object_name,
                        "grasp_count": grasp_count,
                        "filtered_count": filtered_count,
                        "filter_success_rate": filter_success_rate,
                        "stage_handle_detection": True,
                        "stage_joint_axis_analysis": joint_axis_success,
                        "stage_grasp_generation": grasps_success,
                        "stage_grasp_filtering": filtering_success,
                        "step": processed_objects,
                    }
                )
        else:
            failed_objects.append(object_name)
            print(f"   Failed to process {object_name}")

        progress = (i + 1) / len(articulated_objects) * 100
        print(f"   Progress: {progress:.1f}% ({i + 1}/{len(articulated_objects)})")

        # Progress bar similar to run_pipeline.py
        progress_bar_width = 50
        filled_width = int(progress_bar_width * ((i + 1) / len(articulated_objects)))
        progress_bar = "█" * filled_width + "░" * (progress_bar_width - filled_width)

        print(
            f"Progress: [{progress_bar}] {progress:.1f}% ({i + 1}/{len(articulated_objects)})"
        )
        print("=" * 80)

    print("\n" + "=" * 60)
    print("ARTICULATED PIPELINE STAGES 0-3 COMPLETE!")
    print("=" * 60)
    print(
        f"Successfully processed: {processed_objects}/{len(articulated_objects)} objects"
    )
    print(f"Failed objects: {len(failed_objects)}")

    objects_with_joint_analysis = sum(
        1
        for obj in successful_objects
        if obj.get("stages_completed", {}).get("joint_axis_analysis", False)
    )
    objects_with_grasps = sum(
        1
        for obj in successful_objects
        if obj.get("stages_completed", {}).get("grasp_generation", False)
    )
    objects_with_filtered_grasps = sum(
        1
        for obj in successful_objects
        if obj.get("stages_completed", {}).get("grasp_filtering", False)
    )
    print(f"Objects with joint analysis: {objects_with_joint_analysis}")
    print(f"Objects with grasps generated: {objects_with_grasps}")
    print(f"Objects with filtered grasps: {objects_with_filtered_grasps}")

    if failed_objects:
        print(f"\nFailed objects: {', '.join(failed_objects)}")

    summary = {
        "pipeline_stages": "Stages 0-3 - Handle Detection, Joint Analysis, Grasp Generation & Filtering",
        "timestamp": datetime.now().isoformat(),
        "total_objects": len(articulated_objects),
        "processed_objects": processed_objects,
        "objects_with_joint_analysis": objects_with_joint_analysis,
        "objects_with_grasps": objects_with_grasps,
        "objects_with_filtered_grasps": objects_with_filtered_grasps,
        "failed_objects": failed_objects,
        "successful_objects": successful_objects,
    }

    summary_path = os.path.join(output_base, "pipeline_summary_stages0123.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPipeline summary saved to: {summary_path}")

    # Log final results to wandb
    if USE_WANDB:
        wandb.log(
            {
                "pipeline_complete": True,
                "total_processed": processed_objects,
                "total_failed": len(failed_objects),
                "objects_with_joint_analysis": objects_with_joint_analysis,
                "objects_with_grasps": objects_with_grasps,
                "objects_with_filtered_grasps": objects_with_filtered_grasps,
                "success_rate": (processed_objects - len(failed_objects))
                / processed_objects
                * 100
                if processed_objects > 0
                else 0,
            }
        )

        # Create summary table for wandb
        summary_data = []
        for obj in successful_objects:
            stages = obj.get("stages_completed", {})
            summary_data.append(
                [
                    obj["name"],
                    "✓" if stages.get("handle_detection", False) else "✗",
                    "✓" if stages.get("joint_axis_analysis", False) else "✗",
                    "✓" if stages.get("grasp_generation", False) else "✗",
                    "✓" if stages.get("grasp_filtering", False) else "✗",
                ]
            )

        table = wandb.Table(
            columns=[
                "Object",
                "Handle Detection",
                "Joint Analysis",
                "Grasp Generation",
                "Grasp Filtering",
            ],
            data=summary_data,
        )
        wandb.log({"processing_summary": table})

        wandb.finish()
        print("Results logged to Weights & Biases!")
    else:
        print("Pipeline completed (wandb was disabled)")

    print("\nNEXT STEPS:")
    print(
        "1. Review generated handle meshes, joint analysis, grasps, and filtered grasps in output_articulate/"
    )
    print("2. Implement Stage 4: Articulation motion planning")
    print("3. Implement Stage 5: MuJoCo validation with joint constraints")

    return 0 if processed_objects > 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
