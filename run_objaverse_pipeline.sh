#!/bin/bash

# Change to the script directory to ensure relative paths work
cd /root/yejink/thor-grasp

# Set default data path if no argument provided
DATA_PATH=${1:-"/root/mujoco-thor/assets/scenes/procthor-100k-debug/"}

#echo "Using data path: $DATA_PATH"
#ls -l "$DATA_PATH"

#pip install
/opt/miniconda3/envs/mjgrasp/bin/pip install -r requirements.txt

# get all objects from objaverse
/opt/miniconda3/envs/mjgrasp/bin/python scripts/find_objaverse_objects.py "$DATA_PATH" --output objaverse_matched_objs.json

# run the pipeline
/opt/miniconda3/envs/mjgrasp/bin/python run_pipeline.py 


