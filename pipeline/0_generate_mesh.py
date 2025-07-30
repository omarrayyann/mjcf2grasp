import argparse
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import trimesh.transformations as tra
import os
from pathlib import Path


def quaternion_to_matrix(quat):
    if len(quat) == 4:
        w, x, y, z = quat
    else:
        raise ValueError("Quaternion must have 4 elements")

    return tra.quaternion_matrix([w, x, y, z])


def parse_mujoco_xml(xml_path):
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
                            "geom_name": geom.get(
                                "name", f"geom_{len(mesh_instances)}"
                            ),
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


def combine_meshes_to_obj(xml_path, output_path, include_visual_only=True):
    print(f"Parsing MuJoCo XML: {xml_path}")
    mesh_instances, xml_dir = parse_mujoco_xml(xml_path)

    if not mesh_instances:
        print("No mesh instances found in XML file")
        return

    print(f"Found {len(mesh_instances)} mesh instances")

    transformed_meshes = []
    for i, mesh_info in enumerate(mesh_instances):
        print(
            f"Processing mesh {i + 1}/{len(mesh_instances)}: {mesh_info['geom_name']}"
        )

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


def main():
    parser = argparse.ArgumentParser(
        description="Combine meshes from MuJoCo XML into single OBJ"
    )
    parser.add_argument("xml_file", help="Path to MuJoCo XML file")
    parser.add_argument("output_obj", help="Output OBJ file path")
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

    output_path = Path(args.output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        combine_meshes_to_obj(
            xml_path, output_path, include_visual_only=not args.include_collision
        )
        print("Success!")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
