#!/bin/bash

# Change to the script directory to ensure relative paths work
cd /root/yejink/thor-grasp

# Set default data path if no argument provided
#DATA_PATH=${1:-"/root/mujoco-thor/assets/scenes/procthor-100k-debug/"}
DATA_PATH=${1:-"/root/yejink/thor-grasp/assets/objaverse/"}

#echo "Using data path: $DATA_PATH"
#ls -l "$DATA_PATH"

#echo "Get current working directory from root directory"
#pwd


/opt/miniconda3/envs/mjgrasp/bin/pip install -r requirements.txt

# get all objects from objaverse -- this already exists in the directory 
#/opt/miniconda3/envs/mjgrasp/bin/python scripts/find_objaverse_objects.py "$DATA_PATH" --output objaverse_matched_objs.json

# run the pipeline
/opt/miniconda3/envs/mjgrasp/bin/python run_pipeline.py 
