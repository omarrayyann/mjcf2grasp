#!/bin/bash

# Change to the script directory to ensure relative paths work
cd /root/yejink/mjcf2grasp

# Set default data path if no argument provided
#DATA_PATH=${1:-"/root/mujoco-thor/assets/scenes/procthor-100k-debug/"}
DATA_PATH=${1:-"/root/datasets/mujoco-thor/assets/objects/objaverse/"}
START_IDX=${2:-0}
END_IDX=${3:-""}    
#echo "Using data path: $DATA_PATH"
#ls -l "$DATA_PATH"

#echo "Get current working directory from root directory"
#pwd


# Create or update conda environment from YAML file
/opt/miniconda3/bin/conda env update -f conda_env.yaml --prune

# get all objects from objaverse -- this already exists in the directory 
#/opt/miniconda3/envs/mjgrasp/bin/python scripts/find_objaverse_objects.py "$DATA_PATH" --output objaverse_matched_objs.json

# run the pipeline
# Usage: ./run_objaverse_pipeline.sh [DATA_PATH] [START_IDX] [END_IDX]
/opt/miniconda3/envs/mjgrasp/bin/python run_static.py --start $START_IDX --end $END_IDX
