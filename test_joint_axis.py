#!/usr/bin/env python3
"""
Test script to verify joint axis analysis works on a single object
"""
import subprocess
import os
import json

def test_joint_axis_analysis():
    xml_file = "/home/lambda1/Documents/thor-grasp/assets/Thor-Assets/ManipulaTHOR Objects/Doorways/Prefabs/Doorway_10/Doorway_10.xml"
    output_path = "/home/lambda1/Documents/thor-grasp/output_articulate/Doorway_10/test_joint_axis.json"
    
    if not os.path.exists(xml_file):
        print(f"Error: XML file not found at {xml_file}")
        return False
    
    print(f"Testing joint axis analysis on: {xml_file}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        result = subprocess.run([
            "python", "pipeline_articulate/1_find_axis_joint.py",
            xml_file,
            output_path,
            "--object_name", "Doorway_10"
        ], check=True, capture_output=True, text=True)
        
        print("Joint axis analysis completed successfully!")
        print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        
        if os.path.exists(output_path):
            print(f"Output file created: {output_path}")
            with open(output_path, 'r') as f:
                joint_data = json.load(f)
            
            primary_joint = joint_data.get('primary_joint')
            analysis = joint_data.get('analysis', {})
            all_joints = joint_data.get('all_joints', [])
            
            print(f"Total joints found: {len(all_joints)}")
            if primary_joint:
                print(f"Primary joint: {primary_joint['name']}")
                print(f"Joint type: {primary_joint['type']}")
                print(f"Joint axis: {primary_joint['axis']}")
                print(f"Joint range: {primary_joint.get('range', 'unlimited')}")
                print(f"Confidence: {analysis.get('confidence', 'unknown')}")
            else:
                print("No primary joint identified")
                print(f"Error: {analysis.get('error', 'Unknown error')}")
            
            return True
        else:
            print("Error: Output file was not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error in joint axis analysis: {e}")
        print("STDERR:", e.stderr)
        print("STDOUT:", e.stdout)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_joint_axis_analysis()
