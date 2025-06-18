#!/bin/bash

# Check if object name was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <object_name_without_extension>"
  exit 1
fi

OBJECT_NAME="$1"

# Define paths
INPUT_PATH="/home/lambda1/Documents/thor-grasp/models/${OBJECT_NAME}.obj"
TEMP_PATH="/home/lambda1/Documents/thor-grasp/models/test.obj"
OUTPUT_PATH="/home/lambda1/Documents/thor-grasp/models/model.obj"

# Run commands
cd Manifold/build
./manifold "$INPUT_PATH" "$TEMP_PATH" -s
./simplify -i "$TEMP_PATH" -o "$OUTPUT_PATH" -m -r 0.02
cd ..
cd ..
python nvidia.py --object_file models/model.obj --systematic_sampling --quality antipodal
python viz_nvidia.py
