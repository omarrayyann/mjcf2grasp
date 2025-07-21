#!/usr/bin/env python3
"""
Test script to verify grasp generation works on a single object
"""
import subprocess
import os

def test_grasp_generation():
    handle_mesh = "/home/lambda1/Documents/thor-grasp/output_articulate/Doorway_10/Doorway_10_handles.obj"
    output_path = "/home/lambda1/Documents/thor-grasp/output_articulate/Doorway_10/test_grasps.json"
    
    if not os.path.exists(handle_mesh):
        print(f"Error: Handle mesh not found at {handle_mesh}")
        return False
    
    print(f"Testing grasp generation on: {handle_mesh}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        result = subprocess.run([
            "python", "pipeline_articulate/1_generate_grasps.py",
            "--object_file", handle_mesh,
            "--output", output_path,
            "--gripper", "panda",
            "--num_samples", "1000",  # Small number for testing
            "--quality", "antipodal",
            "--min_quality", "0.01",
            "--systematic_sampling",
            "--classname", "articulated_handle",
            "--dataset", "thor_articulated"
        ], check=True, capture_output=True, text=True)
        
        print("Grasp generation completed successfully!")
        print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        
        if os.path.exists(output_path):
            print(f"Output file created: {output_path}")
            import json
            with open(output_path, 'r') as f:
                grasps = json.load(f)
            print(f"Number of grasps generated: {len(grasps.get('transforms', []))}")
            return True
        else:
            print("Error: Output file was not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error in grasp generation: {e}")
        print("STDERR:", e.stderr)
        print("STDOUT:", e.stdout)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_grasp_generation()
