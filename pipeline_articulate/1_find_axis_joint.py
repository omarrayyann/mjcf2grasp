#!/usr/bin/env python3
"""
Stage 3: Find Main Articulation Joint Axis

This script uses LLM analysis to identify the primary joint that should be articulated
to achieve the most significant movement for the object (e.g., door hinge, drawer slide).

The LLM analyzes the MuJoCo XML structure to:
1. Identify all joints in the object
2. Determine which joint is the primary articulation axis
3. Extract joint properties (axis, range, type)
4. Save joint information for motion planning
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh.transformations as tra

# OpenAI API for LLM analysis
try:
    import openai
except ImportError:
    print("Error: OpenAI package not found. Install with: pip install openai")
    sys.exit(1)

def quaternion_to_matrix(quat):
    """Convert quaternion to transformation matrix"""
    if len(quat) == 4:
        # MuJoCo format: w, x, y, z
        w, x, y, z = quat
    else:
        raise ValueError("Quaternion must have 4 elements")
    
    return tra.quaternion_matrix([w, x, y, z])

def setup_openai():
    """Setup OpenAI client with API key"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        # Use default API key if environment variable not set
        api_key = "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"
    
    return openai.OpenAI(api_key=api_key)

def extract_joint_info_from_xml(xml_path):
    """Extract all joint information from the XML file with global transformations"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        joints = []
        
        def parse_body_for_joints(body_elem, parent_transform=np.eye(4), body_path=""):
            """Recursively parse bodies and extract joint information with global transforms"""
            
            # Get body transformation (same logic as mesh extraction)
            pos = body_elem.get("pos", "0 0 0")
            quat = body_elem.get("quat", "1 0 0 0")
            
            pos_values = [float(x) for x in pos.split()]
            pos_matrix = tra.translation_matrix(pos_values)
            
            quat_values = [float(x) for x in quat.split()]
            quat_matrix = quaternion_to_matrix(quat_values)
            
            # Calculate global transformation for this body
            body_transform = np.dot(parent_transform, np.dot(pos_matrix, quat_matrix))
            
            # Current body path for hierarchy tracking
            body_name = body_elem.get('name', 'unnamed_body')
            current_path = f"{body_path}/{body_name}" if body_path else body_name
            
            # Find joints in this body
            for joint in body_elem.findall('joint'):
                # Basic joint information
                joint_info = {
                    'name': joint.get('name', 'unnamed'),
                    'type': joint.get('type', 'unknown'),
                    'axis': joint.get('axis', '0 0 1'),
                    'range': joint.get('range', None),
                    'pos': joint.get('pos', '0 0 0'),
                    'limited': joint.get('limited', 'false'),
                    'damping': joint.get('damping', '0'),
                    'frictionloss': joint.get('frictionloss', '0'),
                    'parent_body': body_name,
                    'body_hierarchy': current_path
                }
                
                # Get joint's local position and transform to global
                joint_pos = joint.get('pos', '0 0 0')
                joint_pos_values = [float(x) for x in joint_pos.split()]
                joint_pos_matrix = tra.translation_matrix(joint_pos_values)
                
                # Global joint transform = body_transform * joint_local_transform
                global_joint_transform = np.dot(body_transform, joint_pos_matrix)
                
                # Extract global position
                global_position = global_joint_transform[:3, 3]
                joint_info['position'] = {  # Using 'position' for LLM compatibility
                    'x': float(global_position[0]),
                    'y': float(global_position[1]),
                    'z': float(global_position[2])
                }
                
                # Transform joint axis to global coordinates
                axis_str = joint.get('axis', '0 0 1')
                local_axis = np.array([float(x) for x in axis_str.split()])
                # Normalize the axis
                if np.linalg.norm(local_axis) > 0:
                    local_axis = local_axis / np.linalg.norm(local_axis)
                
                # Transform axis by the body's rotation (without translation)
                global_axis = body_transform[:3, :3].dot(local_axis)
                joint_info['rotation_axis'] = {  # Using 'rotation_axis' for LLM compatibility
                    'x': float(global_axis[0]),
                    'y': float(global_axis[1]),
                    'z': float(global_axis[2])
                }
                
                # Store both local and global for completeness
                joint_info['local_position'] = {
                    'x': joint_pos_values[0] if len(joint_pos_values) > 0 else 0.0,
                    'y': joint_pos_values[1] if len(joint_pos_values) > 1 else 0.0,
                    'z': joint_pos_values[2] if len(joint_pos_values) > 2 else 0.0
                }
                
                joint_info['local_axis'] = {
                    'x': float(local_axis[0]),
                    'y': float(local_axis[1]),
                    'z': float(local_axis[2])
                }
                
                # Store body transformation info for reference
                joint_info['parent_position'] = {  # Using 'parent_position' for LLM compatibility
                    'x': float(body_transform[0, 3]),
                    'y': float(body_transform[1, 3]),
                    'z': float(body_transform[2, 3])
                }
                
                # Extract rotation from transformation matrix
                body_quaternion = tra.quaternion_from_matrix(body_transform)
                joint_info['parent_rotation'] = {  # Using 'parent_rotation' for LLM compatibility
                    'w': float(body_quaternion[0]),
                    'x': float(body_quaternion[1]),
                    'y': float(body_quaternion[2]),
                    'z': float(body_quaternion[3])
                }
                
                joints.append(joint_info)
            
            # Recursively process child bodies
            for child_body in body_elem.findall('body'):
                parse_body_for_joints(child_body, body_transform, current_path)
        
        # Start parsing from worldbody
        worldbody = root.find('worldbody')
        if worldbody is not None:
            # Process direct children of worldbody
            for body in worldbody.findall('body'):
                parse_body_for_joints(body)
            
            # Also check for joints directly in worldbody
            for joint in worldbody.findall('joint'):
                joint_info = {
                    'name': joint.get('name', 'unnamed'),
                    'type': joint.get('type', 'unknown'),
                    'axis': joint.get('axis', '0 0 1'),
                    'range': joint.get('range', None),
                    'pos': joint.get('pos', '0 0 0'),
                    'limited': joint.get('limited', 'false'),
                    'damping': joint.get('damping', '0'),
                    'frictionloss': joint.get('frictionloss', '0'),
                    'parent_body': 'worldbody',
                    'body_hierarchy': 'worldbody'
                }
                
                # For joints in worldbody, position is already global
                joint_pos = joint.get('pos', '0 0 0')
                joint_pos_values = [float(x) for x in joint_pos.split()]
                joint_info['position'] = {
                    'x': joint_pos_values[0] if len(joint_pos_values) > 0 else 0.0,
                    'y': joint_pos_values[1] if len(joint_pos_values) > 1 else 0.0,
                    'z': joint_pos_values[2] if len(joint_pos_values) > 2 else 0.0
                }
                joint_info['local_position'] = joint_info['position'].copy()
                
                # For joints in worldbody, axis is already global
                axis_str = joint.get('axis', '0 0 1')
                axis_values = [float(x) for x in axis_str.split()]
                joint_info['rotation_axis'] = {
                    'x': axis_values[0] if len(axis_values) > 0 else 0.0,
                    'y': axis_values[1] if len(axis_values) > 1 else 0.0,
                    'z': axis_values[2] if len(axis_values) > 2 else 1.0
                }
                joint_info['local_axis'] = joint_info['rotation_axis'].copy()
                
                joint_info['parent_position'] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                joint_info['parent_rotation'] = {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
                
                joints.append(joint_info)
        
        return joints
    
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return []
    except Exception as e:
        print(f"Error reading XML file: {e}")
        return []

def analyze_joints_with_llm(client, joints, object_name):
    """Use LLM to analyze joints and identify the primary articulation axis"""
    
    if not joints:
        return None, "No joints found in XML"
    
    # Create a detailed description of all joints
    joints_description = []
    for i, joint in enumerate(joints):
        desc = f"Joint {i+1}:\n"
        desc += f"  - Name: {joint['name']}\n"
        desc += f"  - Type: {joint['type']}\n"
        desc += f"  - Axis: {joint['axis']} (x={joint['rotation_axis']['x']}, y={joint['rotation_axis']['y']}, z={joint['rotation_axis']['z']})\n"
        desc += f"  - Position: {joint['pos']} (x={joint['position']['x']}, y={joint['position']['y']}, z={joint['position']['z']})\n"
        desc += f"  - Range: {joint['range'] if joint['range'] else 'unlimited'}\n"
        desc += f"  - Limited: {joint['limited']}\n"
        desc += f"  - Parent body: {joint.get('parent_body', 'unknown')}\n"
        if 'parent_position' in joint:
            desc += f"  - Parent position: x={joint['parent_position']['x']}, y={joint['parent_position']['y']}, z={joint['parent_position']['z']}\n"
        if 'parent_rotation' in joint:
            desc += f"  - Parent rotation: w={joint['parent_rotation']['w']}, x={joint['parent_rotation']['x']}, y={joint['parent_rotation']['y']}, z={joint['parent_rotation']['z']}\n"
        joints_description.append(desc)
    
    joints_text = "\n".join(joints_description)
    
    prompt = f"""
You are analyzing a MuJoCo XML file for an articulated object named "{object_name}".
Your task is to identify the PRIMARY joint that should be articulated to achieve the most significant and useful movement.

Here are all the joints found in the object:

{joints_text}

Based on the object name "{object_name}" and the joint information provided, please:

1. Identify which joint is the PRIMARY ARTICULATION AXIS (the main joint that enables the object's primary function)
2. Explain your reasoning for this choice
3. Provide the joint's key properties

For example:
- For doors: The door hinge joint that allows opening/closing
- For drawers: The slide joint that allows pulling out/pushing in
- For cabinets: The hinge joint for the door/drawer
- For valves: The rotational joint for turning

Respond in JSON format:
{{
    "primary_joint": {{
        "name": "exact_joint_name_from_list",
        "type": "joint_type",
        "axis": "axis_values", 
        "range": "range_values_or_null",
        "reasoning": "explanation_for_choice"
    }},
    "confidence": "high/medium/low",
    "alternative_joints": ["list_of_other_important_joints"],
    "movement_description": "description_of_expected_movement"
}}

If no suitable primary joint can be identified, return:
{{
    "primary_joint": null,
    "confidence": "low",
    "reasoning": "explanation_why_no_joint_identified"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in robotics and articulated object analysis. Analyze joint configurations to identify primary articulation axes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()
        elif "```" in result_text:
            json_start = result_text.find("```") + 3
            json_end = result_text.rfind("```")
            result_text = result_text[json_start:json_end].strip()
        
        try:
            analysis_result = json.loads(result_text)
            return analysis_result, None
        except json.JSONDecodeError as e:
            return None, f"Failed to parse LLM response as JSON: {e}\nResponse: {result_text}"
            
    except Exception as e:
        return None, f"Error calling OpenAI API: {e}"

def find_joint_in_list(joint_name, joints_list):
    """Find the complete joint information by name"""
    for joint in joints_list:
        if joint['name'] == joint_name:
            return joint
    return None

def main():
    parser = argparse.ArgumentParser(
        description='Find primary articulation joint axis using LLM analysis'
    )
    parser.add_argument('xml_file', help='Path to the MuJoCo XML file')
    parser.add_argument('output_path', help='Path to save joint axis information')
    parser.add_argument('--object_name', help='Object name for context (default: infer from XML path)')
    
    args = parser.parse_args()
    
    # Setup OpenAI
    client = setup_openai()
    
    # Determine object name
    if args.object_name:
        object_name = args.object_name
    else:
        # Infer from XML file path
        object_name = Path(args.xml_file).stem
    
    print(f"Analyzing articulation joints for: {object_name}")
    print(f"XML file: {args.xml_file}")
    
    # Extract joint information from XML
    print("Extracting joint information from XML...")
    joints = extract_joint_info_from_xml(args.xml_file)
    
    if not joints:
        print("No joints found in XML file")
        # Save empty result
        result = {
            "object_name": object_name,
            "source_xml": args.xml_file,
            "primary_joint": None,
            "all_joints": [],
            "analysis": {
                "error": "No joints found in XML file",
                "confidence": "low"
            }
        }
    else:
        print(f"Found {len(joints)} joints in XML")
        
        # Analyze with LLM
        print("Analyzing joints with LLM...")
        analysis_result, error = analyze_joints_with_llm(client, joints, object_name)
        
        if error:
            print(f"Error in LLM analysis: {error}")
            result = {
                "object_name": object_name,
                "source_xml": args.xml_file,
                "primary_joint": None,
                "all_joints": joints,
                "analysis": {
                    "error": error,
                    "confidence": "low"
                }
            }
        else:
            # Get complete joint information for the primary joint
            primary_joint_info = None
            if analysis_result.get('primary_joint') and analysis_result['primary_joint'].get('name'):
                primary_joint_info = find_joint_in_list(
                    analysis_result['primary_joint']['name'], joints
                )
            
            result = {
                "object_name": object_name,
                "source_xml": args.xml_file,
                "primary_joint": primary_joint_info,
                "all_joints": joints,
                "analysis": analysis_result
            }
            
            if primary_joint_info:
                print(f"Primary joint identified: {primary_joint_info['name']}")
                print(f"Joint type: {primary_joint_info['type']}")
                print(f"Local axis: {primary_joint_info['axis']}")
                print(f"Global axis: (x={primary_joint_info['rotation_axis']['x']:.4f}, y={primary_joint_info['rotation_axis']['y']:.4f}, z={primary_joint_info['rotation_axis']['z']:.4f})")
                print(f"Local position: {primary_joint_info['pos']}")
                print(f"Global position: (x={primary_joint_info['position']['x']:.4f}, y={primary_joint_info['position']['y']:.4f}, z={primary_joint_info['position']['z']:.4f})")
                print(f"Joint range: {primary_joint_info.get('range', 'unlimited')}")
                if 'parent_position' in primary_joint_info:
                    pp = primary_joint_info['parent_position']
                    print(f"Parent body global position: (x={pp['x']:.4f}, y={pp['y']:.4f}, z={pp['z']:.4f})")
                if 'parent_rotation' in primary_joint_info:
                    pr = primary_joint_info['parent_rotation']
                    print(f"Parent body global rotation: (w={pr['w']:.4f}, x={pr['x']:.4f}, y={pr['y']:.4f}, z={pr['z']:.4f})")
                print(f"Confidence: {analysis_result.get('confidence', 'unknown')}")
                print(f"Reasoning: {analysis_result.get('primary_joint', {}).get('reasoning', 'No reasoning provided')}")
            else:
                print("No primary joint could be identified")
    
    # Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Joint axis information saved to: {args.output_path}")
    
    return 0 if result.get('primary_joint') else 1

if __name__ == "__main__":
    sys.exit(main())
