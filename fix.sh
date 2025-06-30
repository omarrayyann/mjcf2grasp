

INPUT_PATH="/home/lambda1/Documents/thor-grasp/assets/rum_gripper/meshes/new_left.stl"
TEMP_PATH="/home/lambda1/Documents/thor-grasp/assets/objects/test.stl"
OUTPUT_PATH="/home/lambda1/Documents/thor-grasp/assets/rum_gripper/meshes/new_left_new.stl"

cd external_src/Manifold/build
./manifold "$INPUT_PATH" "$TEMP_PATH" -s
./simplify -i "$TEMP_PATH" -o "$OUTPUT_PATH" -m -r 0.02