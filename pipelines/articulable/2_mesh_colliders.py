import os
import numpy as np
import open3d as o3d
from lxml import etree
from pathlib import Path
from typing import List, Optional


def random_rgba():
    """Generate random RGBA color string for visualization."""
    return " ".join([str(np.random.rand()) for _ in range(4)])



def convert_xml_to_use_mesh_colliders(
    xml_string: str,
    xml_base_path: Optional[str] = None,
    ignore_types: Optional[List[str]] = None,
    dynamic_class: str = "__DYNAMIC_MJT__",
) -> str:
    """
    Convert an XML string from using primitive colliders to mesh colliders.
    
    Args:
        xml_string: The input XML string
        xml_base_path: Base path for resolving relative file paths in the XML
        ignore_types: List of object types to ignore when converting (will keep primitives)
        dynamic_class: The CSS class name to use for dynamic collision geoms
        
    Returns:
        Modified XML string with mesh colliders instead of primitives
    """
    if ignore_types is None:
        ignore_types = [
        # too thin
        "plate",
        "key",
        "cd",
        "book",
        "phone",
        "card",
        "bedsheet",
        "lamp",  # not likely to pickup...
        #"pillow",
        "spoon",
        "fork",
        #"plant",  # not likely to pickup...
        # boxy objects that are better prim description
        "laptop",
        "box",
        "statue", # some have very curvy bottom that it cannot stand
        # furntiure with receptacles with objects on top or inside
        "bed",
        "shelving",
        "table",
        "dresser",
        "desk",
    ]  # bad when using mesh collider
    
    # Parse the XML
    root = etree.fromstring(xml_string.encode('utf-8'))
    
    # Helper function to recursively collect all body elements
    def collect_all_bodies(element):
        bodies = [element]
        for child in element.findall("body"):
            bodies.extend(collect_all_bodies(child))
        return bodies
    
    # Find the worldbody and asset sections
    worldbody = root.find("worldbody")
    asset_section = root.find("asset")
    
    if worldbody is None or asset_section is None:
        return xml_string  # Return unchanged if structure is unexpected
    
    # Keep track of used geom names to ensure uniqueness
    used_geom_names = set()
    # Collect all existing geom names in the XML
    for geom in root.findall(".//geom"):
        name = geom.get("name")
        if name:
            used_geom_names.add(name)
    
    # Get all bodies in the worldbody
    main_bodies = worldbody.findall("body")
    
    for main_body in main_bodies:
        body_name = main_body.attrib.get("name", "")
        
        # Check if this object type should be ignored
        should_ignore = any(ignore_type in body_name.lower() for ignore_type in ignore_types)
        if should_ignore:
            continue
            
        # Collect all bodies in the hierarchy
        body_children = collect_all_bodies(main_body)
        
        for child in body_children:
            # Collect visual and collision geoms
            mesh_to_add = []
            primitive_geoms = []
            
            for geom_xml in child.findall("./geom"):
                geom_class = geom_xml.attrib.get("class", "")
                geom_type = geom_xml.attrib.get("type", "")
                mesh_name = geom_xml.attrib.get("mesh", None)
                
                # Collect visual meshes - be more inclusive in detection
                is_visual = (
                    geom_class in ["__VISUAL_MJT__", "visual"] or
                    (mesh_name is not None and geom_type == "mesh" and geom_class not in ["__DYNAMIC_MJT__", "collision"]) or
                    (mesh_name is not None and geom_class == "")  # geoms without class that have mesh
                )
                
                if is_visual and mesh_name is not None:
                    mesh_to_add.append(mesh_name)
                
                # Collect primitive collision geoms
                elif geom_class in ["__DYNAMIC_MJT__", "collision"] and geom_type != "mesh":
                    primitive_geoms.append(geom_xml)
            
            # Process visual meshes if we have any (don't require primitive colliders)
            if not mesh_to_add:
                continue
            
            print(f"Processing body '{child.get('name', 'unnamed')}' with {len(mesh_to_add)} visual meshes and {len(primitive_geoms)} primitive colliders")
            
            n_mesh_colliders = 0
            
            # Process each visual mesh to find corresponding collision meshes
            for mesh_name in mesh_to_add:
                print(f"  Looking for collision meshes for visual mesh: {mesh_name}")
                # Find the mesh asset
                mesh_xml = asset_section.find(f"mesh[@name='{mesh_name}']")
                if mesh_xml is None:
                    continue
                    
                visual_scale = mesh_xml.attrib.get("scale", "1 1 1")
                mesh_file = mesh_xml.attrib.get("file", "")
                
                if not mesh_file:
                    continue
                
                # Determine base directory for file resolution
                if xml_base_path:
                    base_dir = Path(xml_base_path)
                else:
                    base_dir = Path(".")
                
                # Look for collision mesh files
                mesh_file_path = Path(mesh_file)
                if mesh_file_path.is_absolute():
                    collider_workdir = mesh_file_path.parent
                else:
                    collider_workdir = base_dir / mesh_file_path.parent
                
                # Find collision mesh files
                collision_patterns = ["*_collision_*.obj", "*collision*.obj", "*_col_*.obj"]
                all_obj_files = []
                
                for pattern in collision_patterns:
                    all_obj_files.extend(list(collider_workdir.glob(f"**/{pattern}")))
                
                if len(all_obj_files) == 0:
                    continue
                
                # Process each collision mesh file
                for i, collider_obj_file in enumerate(all_obj_files):
                    try:
                        # Load and validate the mesh
                        mesh = o3d.io.read_triangle_mesh(str(collider_obj_file))
                        
                        # Check if mesh is valid
                        if len(mesh.vertices) < 4:
                            continue
                        
                        # Determine inertia type
                        shellinertia = "true"
                        if mesh.is_watertight():
                            volume = mesh.get_volume()
                            if volume > 1e-14:
                                shellinertia = "false"
                        else:
                            continue  # Skip non-watertight meshes
                        
                        # Create mesh asset name
                        asset_mesh_name = collider_obj_file.name
                        
                        # Check if this mesh file is already referenced by any existing mesh asset
                        existing_mesh = None
                        existing_mesh_by_name = asset_section.find(f"mesh[@name='{asset_mesh_name}']")
                        
                        # Also check if the file path is already referenced by a different mesh name
                        existing_mesh_by_file = None
                        for mesh_asset in asset_section.findall("mesh"):
                            existing_file = mesh_asset.get("file", "")
                            if existing_file and Path(existing_file).name == collider_obj_file.name:
                                existing_mesh_by_file = mesh_asset
                                break
                        
                        existing_mesh = existing_mesh_by_name or existing_mesh_by_file
                        
                        if existing_mesh is not None:
                            # If mesh already exists, use the existing mesh name
                            asset_mesh_name = existing_mesh.get("name")
                        else:
                            # Check if this name conflicts with any existing mesh names
                            if existing_mesh_by_name is not None:
                                base_name = collider_obj_file.stem
                                suffix = collider_obj_file.suffix
                                counter = 1
                                while asset_section.find(f"mesh[@name='{base_name}_col_{counter}{suffix}']") is not None:
                                    counter += 1
                                asset_mesh_name = f"{base_name}_col_{counter}{suffix}"
                        
                        # Calculate relative path
                        if xml_base_path:
                            try:
                                relative_path = collider_obj_file.relative_to(base_dir)
                            except ValueError:
                                relative_path = collider_obj_file
                        else:
                            relative_path = collider_obj_file
                        
                        # Add mesh to assets only if it doesn't already exist
                        if existing_mesh is None:
                            mesh_element = etree.Element(
                                "mesh",
                                name=asset_mesh_name,
                                file=str(relative_path),
                                scale=visual_scale,
                                inertia="shell" if shellinertia == "true" else "legacy",
                            )
                            # Add line break after mesh element for better formatting
                            mesh_element.tail = "\n    "
                            asset_section.append(mesh_element)
                        
                        # Add collision geom to body with unique name
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
                        
                        # Add random color for visualization
                        geom_element.set("rgba", random_rgba())
                        
                        # Add line break before geom for better formatting
                        geom_element.tail = "\n    "
                        
                        child.append(geom_element)
                        n_mesh_colliders += 1
                        
                    except Exception as e:
                        # Skip this collision mesh if there's an error
                        print(f"Warning: Failed to process collision mesh {collider_obj_file}: {e}")
                        continue
            
            # Remove primitive collision geoms if mesh colliders were successfully added
            if n_mesh_colliders > 0:
                for geom_xml in primitive_geoms:
                    child_parent = geom_xml.getparent()
                    if child_parent is not None:
                        child_parent.remove(geom_xml)
    
    # Convert back to string
    return etree.tostring(root, encoding='unicode', pretty_print=True)


def convert_xml_file_to_use_mesh_colliders(
    input_xml_path: str,
    output_xml_path: Optional[str] = None,
    ignore_types: Optional[List[str]] = None,
    dynamic_class: str = "__DYNAMIC_MJT__",
) -> str:
    """
    Convert an XML file from using primitive colliders to mesh colliders.
    
    Args:
        input_xml_path: Path to the input XML file
        output_xml_path: Path for the output XML file (if None, overwrites input)
        ignore_types: List of object types to ignore when converting
        dynamic_class: The CSS class name to use for dynamic collision geoms
        
    Returns:
        Modified XML string
    """
    # Read the input file
    with open(input_xml_path, 'r', encoding='utf-8') as f:
        xml_string = f.read()
    
    # Get the base path for resolving relative paths
    xml_base_path = os.path.dirname(os.path.abspath(input_xml_path))
    
    # Convert the XML
    modified_xml = convert_xml_to_use_mesh_colliders(
        xml_string=xml_string,
        xml_base_path=xml_base_path,
        ignore_types=ignore_types,
        dynamic_class=dynamic_class
    )
    
    # Write the output file
    if output_xml_path is None:
        output_xml_path = input_xml_path
    
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(modified_xml)
    
    return modified_xml


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert XML files from primitive colliders to mesh colliders")
    parser.add_argument("--input", "-i", required=True, help="Input XML file path")
    parser.add_argument("--output", "-o", help="Output XML file path (if not provided, overwrites input)")
    parser.add_argument("--dynamic-class", "-c", default="__DYNAMIC_MJT__", help="CSS class name for dynamic collision geoms")
    parser.add_argument("--ignore-types", nargs="*", help="Object types to ignore when converting (space-separated)")
    
    args = parser.parse_args()
    
    # Convert the XML file
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