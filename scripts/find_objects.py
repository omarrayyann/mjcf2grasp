import os
import json
import argparse

def find_objs_with_matching_subfolder(base_dir):
    result = []

    for root, dirs, files in os.walk(base_dir):
        # Only check files in the current directory, not deeper yet
        for file in files:
            if file.endswith('.obj'):
                obj_name = os.path.splitext(file)[0]
                obj_path = os.path.join(root, file)

                # Check if a folder with the same name exists in this directory
                if obj_name in dirs:
                    full_path = os.path.abspath(obj_path)
                    relative_path = os.path.relpath(full_path, base_dir)
                    result.append({
                        "name": obj_name,
                        "path": relative_path
                    })

    return result

def save_to_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find .obj files with matching subfolder in same directory.")
    parser.add_argument("directory", help="Base directory to search")
    parser.add_argument("--output", default="matched_objs.json", help="Output JSON file name")

    args = parser.parse_args()

    matched_objs = find_objs_with_matching_subfolder(args.directory)
    save_to_json(matched_objs, args.output)

    print(f"Found {len(matched_objs)} matching .obj files. Results saved to {args.output}")
