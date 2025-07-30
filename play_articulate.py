import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def load_articulated_objects():
    matched_file = "matched_objs.json"

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
    print(f"\nMuJoCo Grasp Filtering for {object_name}")
    
    # Generate unique output filename with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if per_joint_grasps_json:
        print(f"   Per-joint grasps JSON: {per_joint_grasps_json}")
        
        # Create unique output filename for per-joint filtering
        output_filename = f"joint_meshes_info_filtered_{timestamp}.json"
        filtered_output_path = os.path.join(output_dir, output_filename)
        
        try:
            print(f"   Filtering per-joint grasps using MuJoCo simulation...")
            subprocess.run(
                [
                    "python",
                    "pipeline_articulate/2_filter_mujoco.py",
                    "--object_name",
                    object_name,
                    "--per_joint_summary_json",
                    per_joint_grasps_json,
                    "--xml_file",
                    xml_file,
                    "--num_workers",
                    "10",
                    "--approach_distance",
                    "0.5",
                    "--approach_steps",
                    "5",
                    "--max_successful",
                    "100",
                    "--render",
                    "--filtered"
                ],
                check=True,
            )
            
            # Check the actual output file (might be the original or the new one)
            summary_path = filtered_output_path if os.path.exists(filtered_output_path) else per_joint_grasps_json
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
        print(f"   XML file: {xml_file}")
        
        # Create unique output filename for regular grasp filtering
        base_name = os.path.splitext(os.path.basename(grasps_path))[0]
        filtered_filename = f"{base_name}_filtered_{timestamp}.json"
        filtered_grasps_path = os.path.join(output_dir, filtered_filename)
        
        try:
            print(f"   Filtering grasps using MuJoCo simulation...")
            subprocess.run(
                [
                    "python",
                    "pipeline_articulate/2_filter_mujoco.py",
                    "--object_name",
                    object_name,
                    "--grasps_path",
                    grasps_path,
                    "--xml_file",
                    xml_file,
                    "--num_workers",
                    "10",
                    "--approach_distance",
                    "0.5",
                    "--approach_steps",
                    "5",
                    "--max_successful",
                    "100",
                    "--output_file",
                    filtered_grasps_path,
                    "--render",
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


def main():
    print("MUJOCO GRASP FILTERING")
    print("=" * 50)
    print("This script filters grasps using MuJoCo simulation")
    print("=" * 50)

    articulated_objects = load_articulated_objects()

    if not articulated_objects:
        print("No articulatable objects found to process!")
        return 1

    output_base = "output_articulate"
    
    processed_objects = 0
    failed_objects = []
    successful_objects = []

    for i, obj in enumerate(articulated_objects):
        object_name = obj["name"]
        
        print(f"\nProcessing {i + 1}/{len(articulated_objects)}: {object_name}")
        print("=" * 60)

        object_output_dir = os.path.join(output_base, object_name)
        
        if not os.path.exists(object_output_dir):
            print(f"   Output directory not found: {object_output_dir}")
            failed_objects.append(object_name)
            continue

        # Check for per-joint grasps first
        joint_meshes_json = os.path.join(object_output_dir, "joint_meshes_info_filtered.json")
        filtering_success = False
        filtered_grasps_path = None

        if os.path.exists(joint_meshes_json):
            # Filter per-joint grasps - always run
            filtering_success, filtered_grasps_path = run_grasp_filtering_stage(
                object_name,
                None,
                obj["xml"],
                object_output_dir,
                per_joint_grasps_json=joint_meshes_json,
            )
        else:
            # Check for regular grasps
            grasps_path = os.path.join(object_output_dir, f"{object_name}_grasps.json")
            if os.path.exists(grasps_path):
                filtering_success, filtered_grasps_path = run_grasp_filtering_stage(
                    object_name, grasps_path, obj["xml"], object_output_dir
                )
            else:
                print(f"   No grasps found for filtering: {grasps_path}")

        if filtering_success:
            processed_objects += 1
            successful_objects.append({
                "name": object_name,
                "filtered_grasps": filtered_grasps_path,
                "xml": obj["xml"],
                "output_dir": object_output_dir,
            })
            print(f"   Successfully filtered grasps for {object_name}")
        else:
            failed_objects.append(object_name)
            print(f"   Failed to filter grasps for {object_name}")

        progress = (i + 1) / len(articulated_objects) * 100
        print(f"   Progress: {progress:.1f}% ({i + 1}/{len(articulated_objects)})")

    print("\n" + "=" * 60)
    print("MUJOCO GRASP FILTERING COMPLETE!")
    print("=" * 60)
    print(f"Successfully filtered: {processed_objects}/{len(articulated_objects)} objects")
    print(f"Failed objects: {len(failed_objects)}")

    if failed_objects:
        print(f"\nFailed objects: {', '.join(failed_objects)}")

    summary = {
        "pipeline_stage": "MuJoCo Grasp Filtering",
        "timestamp": datetime.now().isoformat(),
        "total_objects": len(articulated_objects),
        "processed_objects": processed_objects,
        "failed_objects": failed_objects,
        "successful_objects": successful_objects,
    }

    summary_path = os.path.join(output_base, "mujoco_filtering_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFiltering summary saved to: {summary_path}")

    return 0 if processed_objects > 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nFiltering interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)