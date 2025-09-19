# get all objects from objaverse
/opt/miniconda3/envs/mjgrasp/bin/python scripts/find_objaverse_objects.py ../mujoco-thor/assets/scenes/procthor-100k-debug/ --output objaverse_matched_objs.json

# run the pipeline
/opt/miniconda3/envs/mjgrasp/bin/python run_pipeline.py 


