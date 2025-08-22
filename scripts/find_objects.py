import os
import json
import argparse
import xml.etree.ElementTree as ET

ALL_PICKUP_TYPES_THOR = [
    "pillow",
]


def find_objs_with_matching_subfolder(base_dir):
    result = []
    all_pickup_types = set(ALL_PICKUP_TYPES_THOR)
    all_pickup_types = [item.lower() for item in all_pickup_types]

    for root, dirs, files in os.walk(base_dir):
        for file in files:

            if file.endswith(".obj"):
                obj_name = os.path.splitext(file)[0]
                obj_path = os.path.join(root, file)
                json_filename = obj_name + ".json"
                json_path = os.path.join(root, json_filename)

                if obj_name in dirs and os.path.isfile(json_path):
                    subfolder_path = os.path.join(root, obj_name)
                    found_object = False
                    for pickup_type in all_pickup_types:

                        # object_subnames = obj_name.lower().split("_")

                        if pickup_type.lower() in obj_name.lower():
                            found_object = True
                            break

                        # if any(subname in pickup_type for subname in object_subnames):
                        #     found_object = True
                        #     break
                    if not found_object:
                        print(
                            f"Skipping {obj_name} as it is not a recognized pickup type."
                        )
                        continue

                    for subfile in os.listdir(subfolder_path):
                        if subfile.endswith(".xml") and "old" not in subfile.lower():
                            abs_obj_path = os.path.abspath(obj_path)
                            abs_xml_path = os.path.abspath(
                                os.path.join(subfolder_path, subfile)
                            )
                            abs_json_path = os.path.abspath(json_path)

                            # read xml and ensure it has a <joint tag that is not of type free

                            # tree = ET.parse(abs_xml_path)
                            # xml_root = tree.getroot()
                            # has_valid_joint = False
                            # for joint in xml_root.findall(".//joint"):
                            #     if joint.get("type") != "free":
                            #         has_valid_joint = True
                            #         break

                            # if not has_valid_joint:
                            #     print(f"   Skipping {obj_name} as it has no valid joints.")
                            #     continue

                            result.append({"name": obj_name, "xml": abs_xml_path})
                            break
    
    # shuffle
    import random
    random.shuffle(result)

    return result


def save_to_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find .obj files with matching subfolder, valid .xml, and same-named .json."
    )
    parser.add_argument("directory", help="Base directory to search")
    parser.add_argument(
        "--output", default="matched_objs.json", help="Output JSON file name"
    )

    args = parser.parse_args()

    matched_objs = find_objs_with_matching_subfolder(args.directory)
    save_to_json(matched_objs, args.output)

    print(
        f"Found {len(matched_objs)} matching .obj files with valid .xml and .json. Results saved to {args.output}"
    )
