import os
import json
import argparse

def find_xml_files(base_dir):
    result = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.abspath(os.path.join(root, file))
                obj_name = os.path.basename(root)
                result.append({"name": obj_name, "xml": xml_path})
    
    return result


def save_to_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find all .xml files in the given directory."
    )
    parser.add_argument("directory", help="Base directory to search")
    parser.add_argument(
        "--output",
        default="results/objects_list.json",
        help="Output JSON file name",
    )

    args = parser.parse_args()

    xml_files = find_xml_files(args.directory)
    save_to_json(xml_files, args.output)
