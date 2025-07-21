#!/usr/bin/env python3
"""
Test script to verify the complete articulated pipeline on a single object
"""
import json
import os
import subprocess

def test_complete_pipeline():
    # Test with Doorway_10 which we know has handles
    test_object = None
    
    # Load the matched objects to get a test object
    with open("matched_objs.json", "r") as f:
        data = json.load(f)
    
    # Find Doorway_10 or use the first object
    for obj in data:
        if "Doorway_10" in obj["name"]:
            test_object = obj
            break
    
    if not test_object:
        test_object = data[0]  # Use first object if Doorway_10 not found
    
    object_name = test_object["name"]
    xml_file = test_object["xml"]
    
    print(f"Testing complete pipeline on: {object_name}")
    print(f"XML file: {xml_file}")
    
    output_dir = f"output_articulate/{object_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Stage 0: Handle detection
    handle_mesh = f"{output_dir}/{object_name}_handles.obj"
    if not os.path.exists(handle_mesh):
        print("Running Stage 0: Handle detection...")
        result = subprocess.run([
            "python", "pipeline_articulate/0_generate_mesh.py",
            xml_file, handle_mesh, "--handles-only"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Stage 0 failed: {result.stderr}")
            return False
        print("✓ Stage 0 completed")
    else:
        print("✓ Stage 0 already completed")
    
    # Check if handles were found
    handle_info_path = handle_mesh.replace('.obj', '_handle_info.json')
    if not os.path.exists(handle_info_path):
        print("No handle info found, stopping test")
        return False
        
    with open(handle_info_path, 'r') as f:
        handle_info = json.load(f)
    handles = handle_info.get('identified_handles', [])
    
    if not handles:
        print("No handles identified, stopping test")
        return False
        
    print(f"Found {len(handles)} handle(s): {handles}")
    
    # Stage 1: Grasp generation
    grasps_path = f"{output_dir}/{object_name}_grasps.json"
    if not os.path.exists(grasps_path):
        print("Running Stage 1: Grasp generation...")
        result = subprocess.run([
            "python", "pipeline_articulate/1_generate_grasps.py",
            "--object_file", handle_mesh,
            "--output", grasps_path,
            "--gripper", "panda",
            "--num_samples", "1000",  # Small number for testing
            "--quality", "antipodal",
            "--min_quality", "0.01",
            "--systematic_sampling",
            "--classname", "articulated_handle",
            "--dataset", "thor_articulated"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Stage 1 failed: {result.stderr}")
            return False
        print("✓ Stage 1 completed")
    else:
        print("✓ Stage 1 already completed")
    
    # Check grasp count
    with open(grasps_path, 'r') as f:
        grasps_data = json.load(f)
    num_grasps = len(grasps_data.get('transforms', []))
    print(f"Generated {num_grasps} grasps")
    
    if num_grasps == 0:
        print("No grasps generated, stopping test")
        return False
    
    # Stage 2: Grasp filtering
    filtered_grasps_path = grasps_path.replace(".json", "_filtered.json")
    if not os.path.exists(filtered_grasps_path):
        print("Running Stage 2: Grasp filtering...")
        result = subprocess.run([
            "python", "pipeline_articulate/2_filter_mujoco.py",
            "--object_name", object_name,
            "--grasps_path", grasps_path,
            "--xml_file", xml_file,
            "--num_workers", "2",  # Small number for testing
            "--approach_distance", "0.1",
            "--approach_steps", "500",  # Faster for testing
            "--max_successful", "10",  # Stop early for testing
            "--num_shakes", "1",
            "--shake_magnitude", "0.05",
            "--shake_steps", "100"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Stage 2 failed: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return False
        print("✓ Stage 2 completed")
    else:
        print("✓ Stage 2 already completed")
    
    # Check filtered grasp count
    with open(filtered_grasps_path, 'r') as f:
        filtered_data = json.load(f)
    filtered_count = len(filtered_data.get('transforms', []))
    success_rate = (filtered_count / num_grasps * 100) if num_grasps > 0 else 0
    
    print(f"Filtered grasps: {filtered_count}")
    print(f"Success rate: {success_rate:.1f}%")
    
    print(f"\n✓ Complete pipeline test successful!")
    print(f"Results saved in: {output_dir}")
    return True

if __name__ == "__main__":
    test_complete_pipeline()
