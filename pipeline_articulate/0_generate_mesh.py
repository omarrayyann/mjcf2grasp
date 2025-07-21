import argparse
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import trimesh.transformations as tra
import os
from pathlib import Path
import openai
import json
import re
from typing import List, Dict, Any

# Configure OpenAI (you'll need to set your API key)
openai.api_key = "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"

def quaternion_to_matrix(quat):

    if len(quat) == 4:

        w, x, y, z = quat
    else:
        raise ValueError("Quaternion must have 4 elements")

    return tra.quaternion_matrix([w, x, y, z])


def parse_mujoco_xml(xml_path, handle_geoms_only=False, target_geoms=None):
    """
    Parse MuJoCo XML and extract mesh instances.
    
    Args:
        xml_path: Path to XML file
        handle_geoms_only: If True, only extract geometries identified as handles
        target_geoms: List of specific geometry names to extract (used for handles)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    xml_dir = Path(xml_path).parent

    meshes = {}
    assets = root.find("asset")
    if assets is not None:
        for mesh in assets.findall("mesh"):
            name = mesh.get("name")
            file_path = mesh.get("file")
            scale = mesh.get("scale", "1 1 1")

            scale_values = [float(x) for x in scale.split()]
            if len(scale_values) == 1:
                scale_matrix = np.diag([scale_values[0]] * 3 + [1])
            else:
                scale_matrix = np.diag(scale_values + [1])

            meshes[name] = {"file": file_path, "scale_matrix": scale_matrix}

    mesh_instances = []

    def parse_body(body_elem, parent_transform=np.eye(4)):

        pos = body_elem.get("pos", "0 0 0")
        quat = body_elem.get("quat", "1 0 0 0")

        pos_values = [float(x) for x in pos.split()]
        pos_matrix = tra.translation_matrix(pos_values)

        quat_values = [float(x) for x in quat.split()]
        quat_matrix = quaternion_to_matrix(quat_values)

        body_transform = np.dot(parent_transform, np.dot(pos_matrix, quat_matrix))

        for geom in body_elem.findall("geom"):
            if geom.get("type") == "mesh":
                geom_name = geom.get("name", f"geom_{len(mesh_instances)}")
                
                # Filter based on handle detection if requested
                if handle_geoms_only and target_geoms is not None:
                    if geom_name not in target_geoms:
                        continue
                
                mesh_name = geom.get("mesh")
                if mesh_name in meshes:

                    geom_pos = geom.get("pos", "0 0 0")
                    geom_quat = geom.get("quat", "1 0 0 0")

                    geom_pos_values = [float(x) for x in geom_pos.split()]
                    geom_pos_matrix = tra.translation_matrix(geom_pos_values)

                    geom_quat_values = [float(x) for x in geom_quat.split()]
                    geom_quat_matrix = quaternion_to_matrix(geom_quat_values)

                    final_transform = np.dot(
                        body_transform, np.dot(geom_pos_matrix, geom_quat_matrix)
                    )

                    mesh_instances.append(
                        {
                            "mesh_name": mesh_name,
                            "file": meshes[mesh_name]["file"],
                            "scale_matrix": meshes[mesh_name]["scale_matrix"],
                            "transform": final_transform,
                            "geom_name": geom_name,
                        }
                    )

        for child_body in body_elem.findall("body"):
            parse_body(child_body, body_transform)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody.findall("body"):
            parse_body(body)

    return mesh_instances, xml_dir


def load_and_transform_mesh(mesh_info, xml_dir):

    file_path = xml_dir / mesh_info["file"]

    if not file_path.exists():
        print(f"Warning: Mesh file not found: {file_path}")
        return None

    try:

        mesh = trimesh.load(file_path)

        if isinstance(mesh, trimesh.Scene):

            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

        mesh.apply_transform(mesh_info["scale_matrix"])

        mesh.apply_transform(mesh_info["transform"])

        return mesh

    except Exception as e:
        print(f"Error loading mesh {file_path}: {e}")
        return None


def combine_meshes_to_obj(xml_path, output_path, include_visual_only=True, handles_only=False):
    print(f"Parsing MuJoCo XML: {xml_path}")
    
    handle_geoms = []
    if handles_only:
        print("Analyzing XML with LLM to identify handle components...")
        handle_geoms = analyze_xml_for_handles(xml_path)
        
        if not handle_geoms:
            print("No handle components identified. Aborting mesh generation.")
            return
        
        print(f"LLM identified handle components: {handle_geoms}")
    
    mesh_instances, xml_dir = parse_mujoco_xml(
        xml_path, 
        handle_geoms_only=handles_only, 
        target_geoms=handle_geoms
    )

    if not mesh_instances:
        print("No mesh instances found in XML file")
        return

    print(f"Found {len(mesh_instances)} mesh instances")

    transformed_meshes = []
    for i, mesh_info in enumerate(mesh_instances):
        print(f"Processing mesh {i+1}/{len(mesh_instances)}: {mesh_info['geom_name']}")

        if include_visual_only and "Collider" in mesh_info["geom_name"]:
            print(f"  Skipping collision geometry: {mesh_info['geom_name']}")
            continue

        transformed_mesh = load_and_transform_mesh(mesh_info, xml_dir)
        if transformed_mesh is not None:
            transformed_meshes.append(transformed_mesh)
            print(f"  Loaded and transformed: {mesh_info['file']}")

    if not transformed_meshes:
        print("No valid meshes found to combine")
        return

    print(f"Combining {len(transformed_meshes)} meshes...")
    combined_mesh = trimesh.util.concatenate(transformed_meshes)

    print(f"Exporting to: {output_path}")
    combined_mesh.export(output_path)

    print(f"Combined mesh statistics:")
    print(f"  Vertices: {len(combined_mesh.vertices)}")
    print(f"  Faces: {len(combined_mesh.faces)}")
    print(f"  Bounding box: {combined_mesh.bounds}")
    print(f"  Volume: {combined_mesh.volume:.6f}")
    
    if handles_only:
        print(f"Successfully created handle-only mesh from {len(handle_geoms)} components")
        
        handle_info_path = str(output_path).replace('.obj', '_handle_info.json')
        handle_info = {
            "identified_handles": handle_geoms,
            "mesh_instances_processed": len(mesh_instances),
            "source_xml": str(xml_path),
            "output_mesh": str(output_path)
        }
        
        with open(handle_info_path, 'w') as f:
            json.dump(handle_info, f, indent=2)
        print(f"Handle information saved to: {handle_info_path}")


def analyze_xml_for_handles(xml_path: str) -> List[str]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        xml_structure = extract_xml_structure_for_llm(root)
        prompt = create_handle_detection_prompt(xml_structure, xml_path)
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at analyzing MuJoCo XML files for articulated objects. Your task is to identify handle and graspable components that are used to manipulate joints (like door handles, drawer pulls, cabinet knobs, etc.)."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        handle_names = parse_llm_response(response.choices[0].message.content)
        
        print(f"LLM identified {len(handle_names)} handle components: {handle_names}")
        return handle_names
        
    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        return []


def extract_xml_structure_for_llm(root: ET.Element) -> Dict[str, Any]:
    structure = {
        "joints": [],
        "bodies": [],
        "geometries": [],
        "actuators": []
    }
    
    # Build a map of elements to their parent bodies
    body_map = {}
    for body in root.findall(".//body"):
        body_name = body.get("name", "unnamed")
        for child in body:
            if hasattr(child, 'tag'):
                body_map[child] = body_name
    
    for joint in root.findall(".//joint"):
        parent_body = body_map.get(joint, "unknown")
        
        joint_info = {
            "name": joint.get("name", "unnamed"),
            "type": joint.get("type", "unknown"),
            "parent_body": parent_body,
            "axis": joint.get("axis", ""),
            "range": joint.get("range", "")
        }
        structure["joints"].append(joint_info)
    
    for body in root.findall(".//body"):
        body_name = body.get("name", "unnamed")
        body_info = {
            "name": body_name,
            "geometries": [],
            "child_bodies": []
        }
        
        for geom in body.findall("geom"):
            geom_info = {
                "name": geom.get("name", "unnamed"),
                "type": geom.get("type", "unknown"),
                "mesh": geom.get("mesh", ""),
                "size": geom.get("size", ""),
                "pos": geom.get("pos", "0 0 0")
            }
            # Only include geometries that use mesh files (not primitive colliders)
            if geom_info["type"] == "mesh" and geom_info["mesh"]:
                body_info["geometries"].append(geom_info)
                structure["geometries"].append(geom_info)
        
        for child_body in body.findall("body"):
            body_info["child_bodies"].append(child_body.get("name", "unnamed"))
        
        structure["bodies"].append(body_info)
    
    for actuator in root.findall(".//actuator/*"):
        actuator_info = {
            "name": actuator.get("name", "unnamed"),
            "type": actuator.tag,
            "joint": actuator.get("joint", ""),
            "body": actuator.get("body", "")
        }
        structure["actuators"].append(actuator_info)
    
    return structure


def create_handle_detection_prompt(xml_structure: Dict[str, Any], xml_path: str) -> str:
    object_name = Path(xml_path).stem
    
    prompt = f"""
Analyze this MuJoCo XML structure for an articulated object called "{object_name}" and identify ALL geometry components that serve as handles, knobs, pulls, or graspable parts used to manipulate joints.

IMPORTANT: Only consider geometries with type="mesh" that have actual mesh files (.obj). Ignore primitive colliders (box, sphere, cylinder, etc.).

XML Structure Analysis:
======================

JOINTS:
{json.dumps(xml_structure["joints"], indent=2)}

BODIES AND GEOMETRIES (mesh-based only):
{json.dumps(xml_structure["bodies"], indent=2)}

ACTUATORS:
{json.dumps(xml_structure["actuators"], indent=2)}

TASK:
=====
1. Identify which MESH-BASED geometries are handles, knobs, pulls, levers, or any graspable components
2. Look for patterns like:
   - Names containing: handle, knob, pull, lever, grip, grasp
   - Mesh geometries attached to bodies that have joints
   - Small geometric components (often handles are smaller than main structure)
   - Components positioned on doors, drawers, cabinets for manipulation

3. Consider the object type "{object_name}" and typical handles for such objects
4. IGNORE any primitive colliders - only consider geometries with actual mesh files

RESPONSE FORMAT:
===============
Provide your answer as a JSON list containing ONLY the geometry names (from the "name" field) that use mesh files:

{{"handle_geometries": ["geom_name_1", "geom_name_2", ...]}}

If no clear mesh-based handles are found, return: {{"handle_geometries": []}}

IMPORTANT: 
- Only include geometry names that actually exist in the XML structure above
- Only include geometries with type="mesh" (no primitive shapes)
- Focus on components that a robot would grasp to articulate the object
- Include ALL potential graspable mesh components, even if there are multiple handles
"""
    
    return prompt


def parse_llm_response(response_text: str) -> List[str]:
    print(f"LLM Response:\n{response_text}\n")
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            return parsed.get("handle_geometries", [])
        else:
            list_match = re.findall(r'"([^"]+)"', response_text)
            return list_match
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {response_text}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Combine meshes from MuJoCo XML into single OBJ, with LLM-based handle detection for articulated objects"
    )
    parser.add_argument("xml_file", help="Path to MuJoCo XML file")
    parser.add_argument("output_obj", help="Output OBJ file path")
    parser.add_argument(
        "--include-collision",
        action="store_true",
        help="Include collision geometries (default: visual only)",
    )
    parser.add_argument(
        "--handles-only",
        action="store_true",
        help="Extract only handle/graspable components using LLM analysis",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="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",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    if args.handles_only and not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key required for handle detection.")
        print("Set OPENAI_API_KEY environment variable or use --openai-api-key argument")
        return 1

    xml_path = Path(args.xml_file)
    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}")
        return 1

    output_path = Path(args.output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        combine_meshes_to_obj(
            xml_path, 
            output_path, 
            include_visual_only=not args.include_collision,
            handles_only=args.handles_only
        )
        
        if args.handles_only:
            print("Success! Handle-only mesh created for articulated object grasping.")
        else:
            print("Success! Complete object mesh created.")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
