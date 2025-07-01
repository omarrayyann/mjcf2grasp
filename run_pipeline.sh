#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <object_name_without_extension>"
  exit 1
fi

OBJECT_NAME="$1"

INPUT_PATH="../../../assets/objects/${OBJECT_NAME}.obj"
TEMP_PATH="../../../assets/objects/test.obj"
OUTPUT_PATH="../../../assets/objects/model.obj"

echo "Processing object: $OBJECT_NAME"
cd external_src/Manifold/build
./manifold "$INPUT_PATH" "$TEMP_PATH" -s
./simplify -i "$TEMP_PATH" -o "$OUTPUT_PATH" -m -r 0.02
cd ..
cd ..
cd ..

echo "Generating grasps for object: $OBJECT_NAME"
python pipeline/0_generate_grasps.py --object_file assets/objects/test.obj --quality antipodal --output "output/${OBJECT_NAME}_grasps.json"
export LD_LIBRARY_PATH=myenv/lib/python3.10/site-packages/PySide2/Qt/lib:$LD_LIBRARY_PATH
echo "Generating grasps for object: $OBJECT_NAME at output/output/${OBJECT_NAME}_grasps.json"

echo "Visualizing initial grasps for object: $OBJECT_NAME"
python scripts/visualize.py ${OBJECT_NAME}

echo "Filtering grasps for object: $OBJECT_NAME using MuJoCo"
python pipeline/1_filter_mujoco.py ${OBJECT_NAME} 

echo "Visualizing filtered grasps for object: $OBJECT_NAME"
python scripts/visualize.py ${OBJECT_NAME} --verified