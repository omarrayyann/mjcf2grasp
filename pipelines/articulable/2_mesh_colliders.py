import os
import numpy as np
import open3d as o3d
from lxml import etree
from pathlib import Path
from typing import List, Optional


def random_rgba():
    return " ".join([str(np.random.rand()) for _ in range(4)])



def convert_xml_to_use_mesh_colliders(
    xml_string: str,
    xml_base_path: Optional[str] = None,
    ignore_types: Optional[List[str]] = None,
    dynamic_class: str = "__DYNAMIC_MJT__",
) -> str:
    if ignore_types is None:
        ignore_types = [
        "plate",
        "key",
        "cd",
        "book",
        "phone",
        "card",
        "bedsheet",
        "lamp",
        "spoon",
        "fork",
        "laptop",
        "box",
        "statue",
        "bed",
        "shelving",
        "table",
        "dresser",
        "desk",
    ]
    
    root = etree.fromstring(xml_string.encode('utf-8'))
    
    def collect_all_bodies(element):
        bodies = [element]
        for child in element.findall("body"):
            bodies.extend(collect_all_bodies(child))
        return bodies
    
    worldbody = root.find("worldbody")
    asset_section = root.find("asset")
    
    if worldbody is None or asset_section is None:
        return xml_string
    
    used_geom_names = set()
    for geom in root.findall(".//geom"):
        name = geom.get("name")
        if name:
            used_geom_names.add(name)
    
    main_bodies = worldbody.findall("body")
    
    for main_body in main_bodies:
        body_name = main_body.attrib.get("name", "")
        
        should_ignore = any(ignore_type in body_name.lower() for ignore_type in ignore_types)
        if should_ignore:
            continue
            
        body_children = collect_all_bodies(main_body)
        
        for child in body_children:
            mesh_to_add = []
            primitive_geoms = []
            
            for geom_xml in child.findall("./geom"):
                geom_class = geom_xml.attrib.get("class", "")
                geom_type = geom_xml.attrib.get("type", "")
                mesh_name = geom_xml.attrib.get("mesh", None)
                
                is_visual = (
                    geom_class in ["__VISUAL_MJT__", "visual"] or
                    (mesh_name is not None and geom_type == "mesh" and geom_class not in ["__DYNAMIC_MJT__", "collision"]) or
                    (mesh_name is not None and geom_class == "")
                )
                
                if is_visual and mesh_name is not None:
                    mesh_to_add.append(mesh_name)
                
                elif geom_class in ["__DYNAMIC_MJT__", "collision"] and geom_type != "mesh":
                    primitive_geoms.append(geom_xml)
            
            if not mesh_to_add:
                continue
            
            print(f"Processing body '{child.get('name', 'unnamed')}' with {len(mesh_to_add)} visual meshes and {len(primitive_geoms)} primitive colliders")
            
            n_mesh_colliders = 0
            
            for mesh_name in mesh_to_add:
                print(f"  Looking for collision meshes for visual mesh: {mesh_name}")
                mesh_xml = asset_section.find(f"mesh[@name='{mesh_name}']")
                if mesh_xml is None:
                    continue
                    
                visual_scale = mesh_xml.attrib.get("scale", "1 1 1")
                mesh_file = mesh_xml.attrib.get("file", "")
                
                if not mesh_file:
                    continue
                
                if xml_base_path:
                    base_dir = Path(xml_base_path)
                else:
                    base_dir = Path(".")
                
                mesh_file_path = Path(mesh_file)
                if mesh_file_path.is_absolute():
                    collider_workdir = mesh_file_path.parent
                else:
                    collider_workdir = base_dir / mesh_file_path.parent
                
                collision_patterns = ["*_collision_*.obj", "*collision*.obj", "*_col_*.obj"]
                all_obj_files = []
                
                for pattern in collision_patterns:
                    all_obj_files.extend(list(collider_workdir.glob(f"**/{pattern}")))
                
                if len(all_obj_files) == 0:
                    continue
                
                for i, collider_obj_file in enumerate(all_obj_files):
                    try:
                        mesh = o3d.io.read_triangle_mesh(str(collider_obj_file))
                        
                        if len(mesh.vertices) < 4:
                            continue
                        
                        shellinertia = "true"
                        if mesh.is_watertight():
                            volume = mesh.get_volume()
                            if volume > 1e-14:
                                shellinertia = "false"
                        else:
                            continue
                        
                        asset_mesh_name = collider_obj_file.name
                        
                        existing_mesh = None
                        existing_mesh_by_name = asset_section.find(f"mesh[@name='{asset_mesh_name}']")
                        
                        existing_mesh_by_file = None
                        for mesh_asset in asset_section.findall("mesh"):
                            existing_file = mesh_asset.get("file", "")
                            if existing_file and Path(existing_file).name == collider_obj_file.name:
                                existing_mesh_by_file = mesh_asset
                                break
                        
                        existing_mesh = existing_mesh_by_name or existing_mesh_by_file
                        
                        if existing_mesh is not None:
                            asset_mesh_name = existing_mesh.get("name")
                        else:
                            if existing_mesh_by_name is not None:
                                base_name = collider_obj_file.stem
                                suffix = collider_obj_file.suffix
                                counter = 1
                                while asset_section.find(f"mesh[@name='{base_name}_col_{counter}{suffix}']") is not None:
                                    counter += 1
                                asset_mesh_name = f"{base_name}_col_{counter}{suffix}"
                        
                        if xml_base_path:
                            try:
                                relative_path = collider_obj_file.relative_to(base_dir)
                            except ValueError:
                                relative_path = collider_obj_file
                        else:
                            relative_path = collider_obj_file
                        
                        if existing_mesh is None:
                            mesh_element = etree.Element(
                                "mesh",
                                name=asset_mesh_name,
                                file=str(relative_path),
                                scale=visual_scale,
                                inertia="shell" if shellinertia == "true" else "legacy",
                            )
                            mesh_element.tail = "\n    "
                            asset_section.append(mesh_element)
                        
                        base_geom_name = f"{body_name}_{mesh_name}__MeshCollider_{i}"
                        geom_name = base_geom_name
                        counter = 1
                        while geom_name in used_geom_names:
                            geom_name = f"{base_geom_name}_{counter}"
                            counter += 1
                        used_geom_names.add(geom_name)
                        
                        geom_element = etree.Element(
                            "geom",
                            name=geom_name,
                            type="mesh",
                            mesh=asset_mesh_name,
                            **{"class": dynamic_class}
                        )
                        
                        geom_element.set("rgba", random_rgba())
                        
                        geom_element.tail = "\n    "
                        
                        child.append(geom_element)
                        n_mesh_colliders += 1
                        
                    except Exception as e:
                        print(f"Warning: Failed to process collision mesh {collider_obj_file}: {e}")
                        continue
            
            if n_mesh_colliders > 0:
                for geom_xml in primitive_geoms:
                    child_parent = geom_xml.getparent()
                    if child_parent is not None:
                        child_parent.remove(geom_xml)
    
    return etree.tostring(root, encoding='unicode', pretty_print=True)


def convert_xml_file_to_use_mesh_colliders(
    input_xml_path: str,
    output_xml_path: Optional[str] = None,
    ignore_types: Optional[List[str]] = None,
    dynamic_class: str = "__DYNAMIC_MJT__",
) -> str:
    with open(input_xml_path, 'r', encoding='utf-8') as f:
        xml_string = f.read()
    
    xml_base_path = os.path.dirname(os.path.abspath(input_xml_path))
    
    modified_xml = convert_xml_to_use_mesh_colliders(
        xml_string=xml_string,
        xml_base_path=xml_base_path,
        ignore_types=ignore_types,
        dynamic_class=dynamic_class
    )
    
    if output_xml_path is None:
        output_xml_path = input_xml_path
    
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(modified_xml)
    
    return modified_xml


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert XML files from primitive colliders to mesh colliders")
    parser.add_argument("--input", "-i", required=True, help="Input XML file path")
    parser.add_argument("--output", "-o", help="Output XML file path (if not provided, overwrites input)")
    parser.add_argument("--dynamic-class", "-c", default="__DYNAMIC_MJT__", help="CSS class name for dynamic collision geoms")
    parser.add_argument("--ignore-types", nargs="*", help="Object types to ignore when converting (space-separated)")
    
    args = parser.parse_args()
    
    try:
        modified_xml = convert_xml_file_to_use_mesh_colliders(
            input_xml_path=args.input,
            output_xml_path=args.output,
            ignore_types=args.ignore_types,
            dynamic_class=args.dynamic_class
        )
        
        output_path = args.output if args.output else args.input
        print(f"Successfully converted {args.input} to use mesh colliders.")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error converting XML file: {e}")
        exit(1)