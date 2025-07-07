import os
import json
import argparse

def find_objs_with_matching_subfolder(base_dir):
    result = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.obj'):
                obj_name = os.path.splitext(file)[0]
                obj_path = os.path.join(root, file)

                if obj_name in dirs:
                    subfolder_path = os.path.join(root, obj_name)

                    for subfile in os.listdir(subfolder_path):
                        if subfile.endswith('.xml') and 'old' not in subfile.lower():
                            abs_obj_path = os.path.abspath(obj_path)
                            abs_xml_path = os.path.abspath(os.path.join(subfolder_path, subfile))

                            result.append({
                                "name": obj_name,
                                "path": abs_obj_path,
                                "xml": abs_xml_path
                            })
                            break  # Stop after the first valid .xml file

    return result

def save_to_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find .obj files with matching subfolder and valid .xml inside it.")
    parser.add_argument("directory", help="Base directory to search")
    parser.add_argument("--output", default="matched_objs.json", help="Output JSON file name")

    args = parser.parse_args()

    matched_objs = find_objs_with_matching_subfolder(args.directory)
    save_to_json(matched_objs, args.output)

    print(f"Found {len(matched_objs)} matching .obj files with valid .xml. Results saved to {args.output}")
