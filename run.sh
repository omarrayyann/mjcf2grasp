#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <object_name_without_extension>"
  exit 1
fi

OBJECT_NAME="$1"

INPUT_PATH="/home/lambda1/Documents/thor-grasp/assets/objects/${OBJECT_NAME}.obj"
TEMP_PATH="/home/lambda1/Documents/thor-grasp/assets/objects/test.obj"
OUTPUT_PATH="/home/lambda1/Documents/thor-grasp/assets/objects/model.obj"

cd external_src/Manifold/build
./manifold "$INPUT_PATH" "$TEMP_PATH" -s
./simplify -i "$TEMP_PATH" -o "$OUTPUT_PATH" -m -r 0.02
cd ..
cd ..
cd ..
python main.py --object_file assets/objects/test.obj --quality antipodal --output "output/${OBJECT_NAME}_grasps.json"
export LD_LIBRARY_PATH=/home/lambda1/Documents/thor-grasp/myenv/lib/python3.10/site-packages/PySide2/Qt/lib:$LD_LIBRARY_PATH
python visualize.py ${OBJECT_NAME}