import argparse
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import trimesh.transformations as tra
import os
from pathlib import Path
import json
from typing import List, Dict, Any

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


def extract_joint_info_from_xml(xml_path):
    """
    Extract all joint information from the XML file with global transformations.
    Returns a list of joint dicts.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        joints = []
        def parse_body_for_joints(body_elem, parent_transform=np.eye(4), body_path=""):
            pos = body_elem.get("pos", "0 0 0")
            quat = body_elem.get("quat", "1 0 0 0")
            pos_values = [float(x) for x in pos.split()]
            pos_matrix = tra.translation_matrix(pos_values)
            quat_values = [float(x) for x in quat.split()]
            quat_matrix = quaternion_to_matrix(quat_values)
            body_transform = np.dot(parent_transform, np.dot(pos_matrix, quat_matrix))
            body_name = body_elem.get('name', 'unnamed_body')
            current_path = f"{body_path}/{body_name}" if body_path else body_name
            for joint in body_elem.findall('joint'):
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
                joint_pos = joint.get('pos', '0 0 0')
                joint_pos_values = [float(x) for x in joint_pos.split()]
                joint_pos_matrix = tra.translation_matrix(joint_pos_values)
                global_joint_transform = np.dot(body_transform, joint_pos_matrix)
                global_position = global_joint_transform[:3, 3]
                joint_info['position'] = {
                    'x': float(global_position[0]),
                    'y': float(global_position[1]),
                    'z': float(global_position[2])
                }
                axis_str = joint.get('axis', '0 0 1')
                local_axis = np.array([float(x) for x in axis_str.split()])
                if np.linalg.norm(local_axis) > 0:
                    local_axis = local_axis / np.linalg.norm(local_axis)
                global_axis = body_transform[:3, :3].dot(local_axis)
                joint_info['rotation_axis'] = {
                    'x': float(global_axis[0]),
                    'y': float(global_axis[1]),
                    'z': float(global_axis[2])
                }
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
                joint_info['parent_position'] = {
                    'x': float(body_transform[0, 3]),
                    'y': float(body_transform[1, 3]),
                    'z': float(body_transform[2, 3])
                }
                body_quaternion = tra.quaternion_from_matrix(body_transform)
                joint_info['parent_rotation'] = {
                    'w': float(body_quaternion[0]),
                    'x': float(body_quaternion[1]),
                    'y': float(body_quaternion[2]),
                    'z': float(body_quaternion[3])
                }
                joints.append(joint_info)
            for child_body in body_elem.findall('body'):
                parse_body_for_joints(child_body, body_transform, current_path)
        worldbody = root.find('worldbody')
        if worldbody is not None:
            for body in worldbody.findall('body'):
                parse_body_for_joints(body)
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
                joint_pos = joint.get('pos', '0 0 0')
                joint_pos_values = [float(x) for x in joint_pos.split()]
                joint_info['position'] = {
                    'x': joint_pos_values[0] if len(joint_pos_values) > 0 else 0.0,
                    'y': joint_pos_values[1] if len(joint_pos_values) > 1 else 0.0,
                    'z': joint_pos_values[2] if len(joint_pos_values) > 2 else 0.0
                }
                joint_info['local_position'] = joint_info['position'].copy()
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

def select_primary_joint(joints):
    """
    Select the primary articulation joint using a simple heuristic:
    - If only one non-free joint, use that.
    - If multiple, prefer a joint whose name or type contains 'hinge', 'slide', or 'revolute' (case-insensitive).
    - Otherwise, pick the first joint.
    Returns the joint dict or None.
    """
    if not joints:
        return None
    # Filter out 'free' joints
    non_free = [j for j in joints if j.get('type', '').lower() != 'free']
    if not non_free:
        return None
    if len(non_free) == 1:
        return non_free[0]
    # Prefer by name/type
    preferred = [j for j in non_free if any(x in j.get('name', '').lower() or x in j.get('type', '').lower() for x in ['hinge', 'slide', 'revolute'])]
    if preferred:
        return preferred[0]
    return non_free[0]


def combine_meshes_to_obj(xml_path, output_handles_path, output_full_path, include_visual_only=True):
    print(f"Parsing MuJoCo XML: {xml_path}")
    
    # --- HANDLE-ONLY MESH ---
    print("Analyzing XML structure to identify handle components...")
    handle_geoms = find_handle_geoms_by_joints(xml_path)
    mesh_instances, xml_dir = parse_mujoco_xml(
        xml_path, 
        handle_geoms_only=True, 
        target_geoms=handle_geoms
    )
    transformed_meshes = []
    for i, mesh_info in enumerate(mesh_instances):
        if include_visual_only and "Collider" in mesh_info["geom_name"]:
            continue
        transformed_mesh = load_and_transform_mesh(mesh_info, xml_dir)
        if transformed_mesh is not None:
            transformed_meshes.append(transformed_mesh)
    if transformed_meshes:
        combined_mesh = trimesh.util.concatenate(transformed_meshes)
        combined_mesh.export(output_handles_path)
        print(f"Exported handle mesh to: {output_handles_path}")
    else:
        print("No valid handle meshes found to combine.")

    # --- FULL MESH ---
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mesh_file_map = {}
    assets = root.find("asset")
    if assets is not None:
        for mesh in assets.findall("mesh"):
            mesh_name = mesh.get("name")
            mesh_file = mesh.get("file", "")
            mesh_file_map[mesh_name] = mesh_file
    all_obj_geoms = set()
    for body in root.findall('.//body'):
        for geom in body.findall('geom'):
            if geom.get('type') == 'mesh':
                mesh_name = geom.get('mesh', '')
                mesh_file = mesh_file_map.get(mesh_name, '')
                if mesh_file and mesh_file.lower().endswith('.obj'):
                    geom_name = geom.get('name', mesh_name)
                    all_obj_geoms.add(geom_name)
    mesh_instances, xml_dir = parse_mujoco_xml(
        xml_path,
        handle_geoms_only=True,
        target_geoms=list(all_obj_geoms)
    )
    transformed_meshes = []
    for i, mesh_info in enumerate(mesh_instances):
        if include_visual_only and "Collider" in mesh_info["geom_name"]:
            continue
        transformed_mesh = load_and_transform_mesh(mesh_info, xml_dir)
        if transformed_mesh is not None:
            transformed_meshes.append(transformed_mesh)
    if transformed_meshes:
        combined_mesh = trimesh.util.concatenate(transformed_meshes)
        combined_mesh.export(output_full_path)
        print(f"Exported full mesh to: {output_full_path}")
    else:
        print("No valid full meshes found to combine.")

    # After mesh export, also extract and save joint info
    joints = extract_joint_info_from_xml(xml_path)
    primary_joint = select_primary_joint(joints)
    joint_info_path = str(output_full_path).replace('_full.obj', '_joint_axis.json')
    result = {
        "object_name": Path(xml_path).stem,
        "source_xml": str(xml_path),
        "primary_joint": primary_joint,
        "all_joints": joints
    }
    with open(joint_info_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Joint axis information saved to: {joint_info_path}")


def find_handle_geoms_by_joints(xml_path: str) -> List[str]:
    """
    Find handle geometries by identifying non-free joints and extracting
    mesh geometries from their sibling body elements within the same parent body.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find all non-free joints
        non_free_joints = []
        for joint in root.findall(".//joint"):
            joint_type = joint.get("type", "hinge")  # default is hinge
            if joint_type != "free":
                joint_name = joint.get("name", "unnamed")
                non_free_joints.append(joint)
                print(f"Found non-free joint: {joint_name} (type: {joint_type})")
        
        if not non_free_joints:
            print("No non-free joints found in XML")
            return []
        
        handle_geoms = []
        
        # For each non-free joint, find sibling body elements in the same parent
        for joint in non_free_joints:
            joint_name = joint.get("name", "unnamed")
            
            # Find the parent body/element that contains this joint
            parent_element = None
            for body in root.findall(".//body"):
                if joint in body:
                    parent_element = body
                    break
            
            # If not found in body, check worldbody
            if parent_element is None:
                worldbody = root.find("worldbody")
                if worldbody is not None and joint in worldbody:
                    parent_element = worldbody
            
            if parent_element is None:
                print(f"Warning: Could not find parent element for joint {joint_name}")
                continue
            
            parent_name = parent_element.get("name", "worldbody" if parent_element.tag == "worldbody" else "unnamed")
            print(f"Joint {joint_name} is in element: {parent_name}")
            
            # Find all sibling body elements (body tags that are siblings to the joint)
            sibling_bodies = []
            for child in parent_element:
                if child.tag == "body":
                    sibling_bodies.append(child)
            
            print(f"Found {len(sibling_bodies)} sibling bodies to joint {joint_name}")
            
            # Collect all mesh geometries from sibling bodies
            for sibling_body in sibling_bodies:
                sibling_name = sibling_body.get("name", "unnamed")
                
                # Get all mesh geometries from this sibling body and its descendants
                mesh_geoms = collect_mesh_geoms_from_body(sibling_body)
                
                if mesh_geoms:
                    print(f"  Found {len(mesh_geoms)} mesh geometries in sibling body {sibling_name}: {mesh_geoms}")
                    handle_geoms.extend(mesh_geoms)
        
        # Remove duplicates while preserving order
        unique_handle_geoms = []
        seen = set()
        for geom in handle_geoms:
            if geom not in seen:
                unique_handle_geoms.append(geom)
                seen.add(geom)
        
        print(f"Total unique handle geometries found: {len(unique_handle_geoms)}")
        return unique_handle_geoms
        
    except Exception as e:
        print(f"Error during joint-based handle analysis: {e}")
        return []


def collect_mesh_geoms_from_body(body_elem: ET.Element) -> List[str]:
    """
    Recursively collect all mesh geometry names from a body and its descendants.
    """
    mesh_geoms = []
    
    # Get mesh geometries from this body
    for geom in body_elem.findall("geom"):
        if geom.get("type") == "mesh":
            geom_name = geom.get("name")
            mesh_name = geom.get("mesh")
            if geom_name and mesh_name:  # Only include if both name and mesh are specified
                mesh_geoms.append(geom_name)
    
    # Recursively check child bodies
    for child_body in body_elem.findall("body"):
        child_mesh_geoms = collect_mesh_geoms_from_body(child_body)
        mesh_geoms.extend(child_mesh_geoms)
    
    return mesh_geoms


def main():
    parser = argparse.ArgumentParser(
        description="Combine meshes from MuJoCo XML into handle-only and full OBJ files."
    )
    parser.add_argument("xml_file", help="Path to MuJoCo XML file")
    parser.add_argument("output_prefix", help="Output OBJ file prefix (will create _handles.obj and _full.obj)")
    parser.add_argument(
        "--include-collision",
        action="store_true",
        help="Include collision geometries (default: visual only)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    xml_path = Path(args.xml_file)
    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}")
        return 1

    # Remove any trailing .obj, _handles, or _full from output_prefix
    output_prefix = str(args.output_prefix)
    if output_prefix.endswith('.obj'):
        output_prefix = output_prefix[:-4]
    if output_prefix.endswith('_handles'):
        output_prefix = output_prefix[:-8]
    if output_prefix.endswith('_full'):
        output_prefix = output_prefix[:-5]
    output_handles_path = Path(f"{output_prefix}_handles.obj")
    output_full_path = Path(f"{output_prefix}_full.obj")
    output_handles_path.parent.mkdir(parents=True, exist_ok=True)
    output_full_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        combine_meshes_to_obj(
            xml_path,
            output_handles_path,
            output_full_path,
            include_visual_only=not args.include_collision
        )
        print("Success! Handle-only and full object meshes created.")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())